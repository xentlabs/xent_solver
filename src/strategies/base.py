from dataclasses import dataclass
import torch
from enum import Enum


class SelectionMode(str, Enum):
    TOP_K = "topk"
    EPSILON_TOP_K = "epsilon_topk"
    SOFTMAX = "softmax"


@dataclass
class StrategyParams:
    momentum_alpha: float
    top_k: int


@dataclass
class SelectionParams:
    mode: SelectionMode # {topk, epsilon_topk, softmax}
    epsilon: float           # used when mode == epsilon_topk
    temperature: float       # used when mode == softmax


class OptimizationStrategy:
    """Abstract interface for proposing candidate variable tokens.

    Implementations should update internal state on gradients and propose
    new candidate token IDs given an exploration pool.
    """

    def __init__(self, params: StrategyParams, selection: SelectionParams):
        self.params = params
        self.selection = selection

    def on_gradients(self, grad_batch_tokens: torch.Tensor, gradient_vocab_scores: torch.Tensor):
        """Update internal state from vocab-level gradients.

        Args:
            grad_batch_tokens: [B, L]
            gradient_vocab_scores: [B, L, V]
        """
        raise NotImplementedError

    def select_grad_batch(self, candidates: torch.Tensor, scores: torch.Tensor, limit: int) -> torch.Tensor:
        """Choose a subset of candidate rows for gradient evaluation."""

        if candidates.shape[0] <= limit:
            return candidates
        return candidates[:limit]

    def propose_candidates(self, explore_pool_tokens: torch.Tensor) -> torch.Tensor:
        """Return new candidate tokens from the given explore pool.

        Args:
            explore_pool_tokens: [B, L]
        Returns:
            candidate_tokens: [B, L]
        """
        raise NotImplementedError
