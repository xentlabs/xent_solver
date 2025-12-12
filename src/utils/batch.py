from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.dsl.program import Program
from src.dsl.scorer import ProgramScorer
from src.dsl.initializer import random_flat_variables


@dataclass
class SearchBatchPlan:
    bs_grad: int
    repeat_factor: int

    @property
    def bs_candidates(self) -> int:
        return self.bs_grad * self.repeat_factor


def clean_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_and_measure(fn: Callable[[], None]) -> bool:
    clean_cuda_cache()
    try:
        fn()
        return True
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        clean_cuda_cache()
        return False


def _measure_batch(batch_size: int, scorer: ProgramScorer, program: Program,
                   banned_ids: torch.Tensor, compute_grads: bool) -> None:
    tokens = random_flat_variables(program.vocab_size, program.total_var_len, batch_size, banned_ids, program.device)
    if compute_grads:
        scorer.compute_loss_and_grads(tokens)
    else:
        scorer.compute_loss(tokens)
    del tokens


def _binary_search_max(batch_fn: Callable[[int], bool], start: int) -> int:
    hi = start
    success = batch_fn(hi)
    
    if success:
        while success:
            hi *= 2
            success = batch_fn(hi)
    else:
        while not success:
            hi = hi // 2
            success = batch_fn(hi)
        hi *= 2
    lo = hi // 2

    # Binary search with 15% margin
    while int(lo * 1.15) < hi:
        mid = (lo + hi) // 2
        if batch_fn(mid):
            lo = mid
        else:
            hi = mid

    return lo


def compute_search_batch_plan(
    scorer: ProgramScorer,
    program: Program,
    banned_ids: torch.Tensor,
    fabric,
) -> SearchBatchPlan:
    if not torch.cuda.is_available():
        return SearchBatchPlan(bs_grad=16, repeat_factor=2)

    world = fabric.world_size
    safety_factor = 0.7

    # Find maximum batch sizes
    grad_cap = _binary_search_max(
        lambda batch: _run_and_measure(lambda: _measure_batch(batch, scorer, program, banned_ids, compute_grads=True)),
        start=128
    )
    forward_cap = _binary_search_max(
        lambda batch: _run_and_measure(lambda: _measure_batch(batch, scorer, program, banned_ids, compute_grads=False)),
        start=4 * grad_cap
    )

    # Apply safety factor
    grad_cap = int(grad_cap * safety_factor)
    forward_cap = int(forward_cap * safety_factor)

    # Ensure grad batch fits within forward capacity
    grad_cap = max(1, min(grad_cap, forward_cap // world))
    repeat_factor = forward_cap // (grad_cap * world)

    return SearchBatchPlan(
        bs_grad=max(1, grad_cap),
        repeat_factor=max(1, repeat_factor),
    )