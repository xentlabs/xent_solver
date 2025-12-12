from typing import Tuple

import torch

from src.utils.fabric import FabricPlus
from src.engine.task import TaskEngine
from src.engine.config import OptimizerConfig
from src.utils.batch import SearchBatchPlan
from src.strategies.base import OptimizationStrategy
from src.engine.cache import GradientCache


def _repeat_rows(tensor: torch.Tensor, repeat_factor: int) -> torch.Tensor:
    return tensor.repeat(repeat_factor, *([1] * (tensor.dim() - 1)))


def _keep_unique_first(flat: torch.Tensor, total: torch.Tensor):
    """Keep unique rows, preserving first occurrence.

    - GPU/CPU: uses torch.unique(dim=0) + scatter_reduce('amin') for first indices.
    - MPS: uses a lexicographic-hash + adjacent-compare fallback (no unique_dim). -> WTF?
    """
    B, L = flat.shape
    if B <= 1:
        return flat, total
    if flat.device.type != 'mps':
        # Standard fast path (CUDA/CPU)
        uniq, inverse = torch.unique(flat, dim=0, return_inverse=True)
        row_idx = torch.arange(B, device=flat.device, dtype=torch.long)
        first = torch.full((uniq.shape[0],), B, device=flat.device, dtype=torch.long)
        first = first.scatter_reduce(0, inverse, row_idx, reduce='amin', include_self=False)
        order = torch.argsort(first)
        return uniq[order], total[first[order]]

    # MPS fallback: lexicographic-ish hashing then adjacent dedup
    h = torch.zeros(B, dtype=torch.int64, device=flat.device)
    base = torch.tensor(1000003, dtype=torch.int64, device=flat.device)
    one = torch.tensor(1, dtype=torch.int64, device=flat.device)
    flat_i64 = flat.to(torch.int64)
    for i in range(L):
        h = h * base + (flat_i64[:, i] + one)
    order = torch.argsort(h)
    sorted_flat = flat[order]
    sorted_total = total[order]
    diff = (sorted_flat[1:] != sorted_flat[:-1]).any(dim=1)
    mask = torch.ones(B, dtype=torch.bool, device=flat.device)
    mask[1:] = diff
    return sorted_flat[mask], sorted_total[mask]

def _order_candidates(candidates: torch.Tensor, totals: torch.Tensor):
    order = torch.argsort(totals)
    return candidates[order], totals[order]


class SearchOptimizer:
    def __init__(
        self,
        engine: TaskEngine,
        strategy: OptimizationStrategy,
        config: OptimizerConfig,
        fabric: FabricPlus,
        batch_plan: SearchBatchPlan,
        verbose: bool = False,
    ):
        self.engine = engine
        self.strategy = strategy
        self.cfg = config
        self.annealing_cfg = config.annealing
        self.fabric = fabric
        self.grad_cache = GradientCache()
        self.verbose = bool(verbose)
        self.bs_grad = batch_plan.bs_grad
        self.bs_candidates = batch_plan.bs_candidates
        self.repeat_factor = batch_plan.repeat_factor
        self.temperature = float(self.annealing_cfg.initial_temp) if self.annealing_cfg else 0.0

    def _print(self, msg: str):
        if not self.verbose:
            return
        self.fabric.print(msg)

    def _cool_if_needed(self, stage: str):
        if self.annealing_cfg is None:
            return
        if self.annealing_cfg.schedule != stage:
            return
        cooled = self.temperature * self.annealing_cfg.decay
        self.temperature = float(max(self.annealing_cfg.min_temp, cooled))

    def _accept_mask(self, current: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        better = proposed < current
        if self.annealing_cfg is None:
            return better

        accept = better.clone()
        worse = ~better
        if worse.any():
            delta = proposed[worse] - current[worse]
            probs = torch.exp(-delta / self.temperature)
            accept[worse] = torch.rand_like(probs) < probs
        return accept

    def _improve_candidates(self, tokens: torch.Tensor, scores: torch.Tensor, proposals: torch.Tensor, new_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        accept = self._accept_mask(scores, new_scores)
        if accept.any():
            tokens = torch.where(accept.view(-1, 1), proposals, tokens)
            scores = torch.where(accept, new_scores, scores)
        return tokens, scores

    def _score_with_retry(self, proposals: torch.Tensor, can_change_rf: bool = True) -> torch.Tensor:
        """Score proposals with automatic batch splitting on OOM."""
        try:
            with torch.inference_mode():
                return self.engine.compute_loss(proposals)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise

            if proposals.shape[0] <= 1:
                raise RuntimeError("OOM with batch size 1, cannot split further")
                        
            # Split and process recursively
            mid = proposals.shape[0] // 2
            first_score = self._score_with_retry(proposals[:mid], can_change_rf=False)
            second_score = self._score_with_retry(proposals[mid:], can_change_rf=False)
            
            # Reduce repeat_factor for future iterations (preserves original behavior)
            if can_change_rf:
                self.repeat_factor = max(1, self.repeat_factor - 1)

            return torch.cat([first_score, second_score], dim=0)

    def _perform_candidate_search(self, explore_tokens: torch.Tensor, explore_scores: torch.Tensor, best_loss: float) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        candidate_pat, candidate_step = 0, 0
        current_best_tokens, current_best_loss = None, best_loss
        
        while candidate_pat < self.cfg.candidate_patience:
            candidate_pat += 1
            candidate_step += 1
            new_candidates = self.strategy.propose_candidates(explore_tokens)
            new_scores = self._score_with_retry(new_candidates, can_change_rf=(candidate_step == 1))

            explore_tokens, explore_scores = self._improve_candidates(
                explore_tokens, explore_scores, new_candidates, new_scores
            )
            self._cool_if_needed("candidate")

            updated_tokens, updated_loss = self.fabric.sync_best_from_pool(explore_tokens, explore_scores)
            if updated_loss < current_best_loss:
                current_best_tokens, current_best_loss = updated_tokens, updated_loss
                candidate_pat = 0
                decoded = self.engine.program.decode_vars(current_best_tokens)
                score_repr = self.engine.program.report_score(current_best_loss)
                self._print(f"\tCandidate step {candidate_step} -- batch_size={explore_tokens.shape[0]} -- best_score={score_repr:.4f} -- vars={decoded}")
        
        return explore_tokens, explore_scores, current_best_loss, current_best_tokens

    def run(self, init_tokens: torch.Tensor):
        self._print(f"Batch plan: bs_grad={self.bs_grad} repeat_factor={self.repeat_factor}")
        init_scores = self._score_with_retry(init_tokens, can_change_rf=False)
        candidates, scores = _order_candidates(init_tokens, init_scores)
        best_tokens, best_loss = self.fabric.sync_best_from_pool(candidates, scores)

        best_grad_loss, final_grad_batch = best_loss, None
        grad_pat, grad_step = 0, 0

        while grad_pat < self.cfg.grad_patience:
            grad_step += 1
            grad_pat += 1

            # 1. Gradient Step
            grad_batch = self.strategy.select_grad_batch(candidates, scores, self.bs_grad)
            batch_scores, vocab_grads = self.grad_cache.compute(
                grad_batch, self.engine.compute_loss_and_vocab_grads
            )
            
            self.strategy.on_gradients(grad_batch, vocab_grads)
            gathered_tokens = self.fabric.all_gather_flat(grad_batch)
            gathered_scores = self.fabric.all_gather_flat(batch_scores)

            explore_tokens = _repeat_rows(gathered_tokens, self.repeat_factor)
            explore_scores = gathered_scores.repeat(self.repeat_factor)
        
            # 2. Candidate Search
            explore_tokens, explore_scores, new_best_loss, new_best_tokens = self._perform_candidate_search(
                explore_tokens, explore_scores, best_loss
            )

            # Update global best if improved
            if new_best_loss < best_loss:
                best_tokens, best_loss = new_best_tokens, new_best_loss

            uniq_tokens, uniq_scores = _keep_unique_first(explore_tokens, explore_scores)
            order = torch.argsort(uniq_scores)
            candidates = uniq_tokens[order]
            scores = uniq_scores[order]

            self._cool_if_needed("grad")

            pool_best_tokens, pool_best_loss = self.fabric.sync_best_from_pool(candidates, scores)
            if pool_best_loss < best_loss:
                best_tokens, best_loss = pool_best_tokens, pool_best_loss

            if pool_best_loss < best_grad_loss:
                best_grad_loss = pool_best_loss
                grad_pat = 0

            final_grad_batch = self.strategy.select_grad_batch(candidates, scores, self.bs_grad).clone().detach()

            decoded = self.engine.program.decode_vars(best_tokens)
            score_repr = self.engine.program.report_score(best_loss)
            self._print(f"Grad step {grad_step} -- batch_size={final_grad_batch.shape[0]} -- best_score={score_repr:.4f} -- vars={decoded}")

            self._cool_if_needed("step")

        best_tokens, best_loss = self.fabric.global_sync_best(best_tokens, best_loss)
        return best_tokens, best_loss, final_grad_batch
