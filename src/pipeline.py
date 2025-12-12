import json
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.model import setup_fabric, load_model_and_tokenizer
from src.utils.args import call_parser
from src.utils.batch import compute_search_batch_plan, clean_cuda_cache
from src.utils.logging import ResultAggregator

from src.data.loader import MapDataset
from src.data.constraints import ConstraintPolicy

from src.engine.task import TaskEngine
from src.engine.optimizer import SearchOptimizer
from src.engine.config import OptimizerConfig

from src.strategies.base import StrategyParams
from src.strategies.gcg import GcgStrategy

from src.dsl.template import materialize_spec
from src.dsl.initializer import random_flat_variables

def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def collect_gpu_info(fabric):
    try:
        import torch
        world = fabric.world_size
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "rank": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "capability": f"{props.major}.{props.minor}",
                })
        return {
            "world_size": world,
            "devices": devices,
            "precision": fabric.precision,
            "strategy": fabric.strategy.__class__.__name__ if fabric.strategy else None,
        }
    except Exception:
        return {"world_size": fabric.world_size, "devices": []}


def _execute_task_for_model(fabric, config, task_spec, build_spec_for_map, *, verbose: bool = False):
    model_results = []
    seed_cache = None

    tpl = build_spec_for_map(task_spec.details)
    policy = ConstraintPolicy.from_map_hyperparams(task_spec.hyper_params)

    for m_idx, model_name in enumerate(config["model_names"], start=1):
        if verbose:
            fabric.print(f"  ({m_idx}/{len(config['model_names'])}) {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, fabric)

        # materialize program spec
        spec = materialize_spec(tpl, model, tokenizer, bos=bool(task_spec.hyper_params.get("bos", False)))
        engine = TaskEngine.from_spec(model, tokenizer, spec, model.device)

        # constraints
        banned = policy.banned_token_ids(tokenizer, engine.program)

        # init candidates
        batch_plan = compute_search_batch_plan(
            scorer=engine.scorer,
            program=engine.program,
            banned_ids=banned,
            fabric=fabric,
        )

        init_tokens = random_flat_variables(
            engine.program.vocab_size,
            engine.program.total_var_len,
            batch_plan.bs_candidates,
            banned,
            engine.device,
        )
        if seed_cache is not None:
            num_seed = min(seed_cache.shape[0], init_tokens.shape[0])
            init_tokens[:num_seed] = seed_cache[:num_seed].to(init_tokens.device)

        # strategy and optimizer
        optimizer_cfg = OptimizerConfig.from_settings(config)
        strategy = GcgStrategy(
            StrategyParams(
                momentum_alpha=config["alpha"],
                top_k=config["top_k"],
            ),
            selection=optimizer_cfg.selection,
            banned_token_ids=banned,
        )
        optimizer = SearchOptimizer(
            engine=engine,
            strategy=strategy,
            config=optimizer_cfg,
            fabric=fabric,
            batch_plan=batch_plan,
            verbose=bool(config.get("verbose", False)),
        )

        best_tokens, best_loss, last_batch = optimizer.run(init_tokens)
        # Prepare result entry
        model_results.append({
            "model_name": model_name,
            "goal": engine.program.spec.goal,
            "best_objective": engine.program.report_score(best_loss),
            "best_loss_base2": engine.program.report_score(best_loss) / __import__('math').log(2),
            "best_vars": engine.decode_variables(best_tokens),
            "search_loss_internal": float(best_loss),
            "batch_per_rank": batch_plan.bs_candidates,
            "batch_global": batch_plan.bs_candidates * fabric.world_size,
        })
        seed_cache = last_batch.detach()

        del model, tokenizer, engine, banned, init_tokens, strategy, optimizer
        clean_cuda_cache()

    return model_results


def _run_task_with_repeats(fabric, config, task_spec, build_spec_for_map, repeat_count: int, *, verbose: bool = False):
    runs = []
    minimal = []
    for r in range(1, repeat_count + 1):
        if verbose:
            fabric.print(f"  Repeat {r}/{repeat_count}")
        model_results = _execute_task_for_model(
            fabric=fabric,
            config=config,
            task_spec=task_spec,
            build_spec_for_map=build_spec_for_map,
            verbose=verbose,
        )
        last = model_results[-1] if model_results else {}
        run_final = {
            "variables": last.get("best_vars", {}),
            "loss": last.get("search_loss_internal"),
            "objective": last.get("best_objective"),
        }
        runs.append({"run": r, "models": model_results, "final": run_final})
        minimal.append({"run": r, **run_final})

    minimal_sorted = sorted(minimal, key=lambda x: (float('inf') if x["loss"] is None else float(x["loss"])))
    best_run_id = minimal_sorted[0]["run"] if minimal_sorted else 1
    best_models = next((rr["models"] for rr in runs if rr["run"] == best_run_id), runs[-1]["models"] if runs else [])
    best_final = {k: v for k, v in minimal_sorted[0].items() if k != "run"} if minimal_sorted else {"variables": {}, "loss": None, "objective": None}

    if verbose:
        fabric.print(f"  Best repeat: {best_run_id}/{repeat_count} score={best_final.get('objective')}")

    return runs, minimal_sorted, best_models, best_final


def run_generic(build_spec_for_map, description: str, conf_path: str, data_path: str, output_dir: str, verbose: bool = False):
    conf_path, data_path, output_dir, cli_verbose, gpu_ids = call_parser(description, conf_path, data_path, output_dir)
    verbose = verbose or cli_verbose

    config_path = Path(conf_path)
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    fabric = setup_fabric(config["precision"], devices=gpu_ids, strategy=config["strategy"])
    gpu_info = collect_gpu_info(fabric)

    if verbose:
        config["verbose"] = True
    
    dataset = MapDataset(str(data_path))
    def _identity_collate(batch):
        return batch[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_identity_collate)

    aggregator = ResultAggregator(
        output_dir=output_dir,
        config=config,
        data_path=data_path,
        gpu_info=gpu_info,
        fabric=fabric,
        spec_description=description,
    )

    for idx, item in enumerate(dataloader, start=1):
        task_spec = item
        repeat_count = max(1, int(task_spec.hyper_params.get("repeat", 1)))
        fabric.print(f"\n{'='*60}\n[{idx}/{len(dataset)}] {task_spec.name}\n{'='*60}")

        runs, minimal_sorted, best_models, best_final = _run_task_with_repeats(
            fabric=fabric,
            config=config,
            task_spec=task_spec,
            build_spec_for_map=build_spec_for_map,
            repeat_count=repeat_count,
            verbose=bool(config.get("verbose", False)),
        )

        aggregator.add_task_result(
            index=idx,
            task_spec=task_spec,
            runs=runs,
            minimal_sorted=minimal_sorted,
            best_models=best_models,
            best_final=best_final,
        )
        aggregator.flush()

    aggregator.flush()

