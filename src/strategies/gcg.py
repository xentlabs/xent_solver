import torch

from .base import OptimizationStrategy, StrategyParams, SelectionParams, SelectionMode


class GcgStrategy(OptimizationStrategy):
    def __init__(
        self,
        params: StrategyParams,
        selection: SelectionParams | None = None,
        banned_token_ids: torch.Tensor | None = None,
    ):
        super().__init__(params, selection)
        self._topk_indices = None  # [B, L, K]
        self._banned = banned_token_ids

    def on_gradients(self, grad_batch_tokens: torch.Tensor, gradient_vocab_scores: torch.Tensor):
        scores = gradient_vocab_scores
        if self._banned is not None and self._banned.numel() > 0:
            scores = scores.clone()
            scores.index_fill_(-1, self._banned.to(scores.device), float("inf"))
        self._topk_indices = (-scores).topk(int(self.params.top_k), dim=-1).indices

    def select_grad_batch(self, candidates: torch.Tensor, scores: torch.Tensor, limit: int) -> torch.Tensor:
        if candidates.shape[0] <= limit:
            return candidates

        mode = self.selection.mode
        if mode == SelectionMode.TOP_K:
            return candidates[:limit]

        if mode == SelectionMode.EPSILON_TOP_K:
            elite = max(1, int(round((1.0 - self.selection.epsilon) * limit)))
            elite = min(elite, candidates.shape[0])
            top = candidates[:elite]
            remaining = limit - top.shape[0]
            if remaining <= 0 or elite == candidates.shape[0]:
                return top
            tail = candidates[elite:]
            idx = torch.randint(0, tail.shape[0], (remaining,), device=candidates.device)
            return torch.cat([top, tail[idx]], dim=0)

        if mode == SelectionMode.SOFTMAX:
            weights = torch.softmax(-scores / self.selection.temperature, dim=0)
            count = min(limit, candidates.shape[0])
            idx = torch.multinomial(weights, num_samples=count, replacement=False)
            return candidates[idx]

        return candidates[:limit]

    def propose_candidates(self, explore_pool_tokens: torch.Tensor) -> torch.Tensor:
        if self._topk_indices is None:
            return explore_pool_tokens
        B, L = explore_pool_tokens.shape
        K = self._topk_indices.shape[-1]
        tk = self._topk_indices
        if tk.shape[0] != B:
            repeat_factor = (B + tk.shape[0] - 1) // tk.shape[0]
            tk = tk.repeat(repeat_factor, 1, 1)[:B]
        pos = torch.randint(0, L, (B,), device=explore_pool_tokens.device)
        ch = torch.randint(0, K, (B,), device=explore_pool_tokens.device)
        out = explore_pool_tokens.clone()
        out[torch.arange(B), pos] = tk[torch.arange(B), pos, ch]
        return out
