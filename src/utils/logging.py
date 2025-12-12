from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List
import json


@dataclass
class ResultAggregator:
    """Centralizes persistence of solver run summaries."""

    output_dir: str | Path
    config: dict
    data_path: Path
    gpu_info: Dict[str, Any]
    fabric: Any
    spec_description: str
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.data_path = Path(self.data_path)
        self.started_at = time()

    def add_task_result(
        self,
        index: int,
        task_spec,
        runs: List[Dict[str, Any]],
        minimal_sorted: List[Dict[str, Any]],
        best_models: List[Dict[str, Any]],
        best_final: Dict[str, Any],
    ) -> None:
        """Append a fully materialized task summary."""

        entry = {
            "index": index,
            "name": task_spec.name,
            "details": task_spec.details,
            "hyper_params": task_spec.hyper_params,
            "final": best_final,
            "models": best_models,
            "repeats": runs,
            "results_sorted": [
                {k: v for k, v in result.items() if k != "run"}
                for result in minimal_sorted
            ],
        }
        self.entries.append(entry)

    def _summary_payload(self) -> Dict[str, Any]:
        return {
            "spec_description": self.spec_description,
            "config": self.config,
            "data_path": str(Path(self.data_path).resolve()),
            "gpu": self.gpu_info,
            "total_time_seconds": time() - self.started_at,
            "entries": self.entries,
        }

    def flush(self) -> None:
        if not self.fabric.is_global_zero:
            return

        out = self.path / "summary.json"
        with out.open("w") as f:
            json.dump(self._summary_payload(), f, indent=2)


