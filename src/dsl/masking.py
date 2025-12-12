from dataclasses import dataclass
import torch


@dataclass
class Constraints:
    banned_ids: torch.Tensor
    per_position_mask: torch.Tensor | None = None  # [L, V] True=allowed

    def apply_vocab_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Set disallowed token scores to +inf (lower is better)."""
        if self.banned_ids is not None and self.banned_ids.numel() > 0:
            scores.index_fill_(-1, self.banned_ids.to(scores.device), float('inf'))
        if self.per_position_mask is not None:
            mask = ~self.per_position_mask.to(scores.device)
            scores = scores.masked_fill(mask.unsqueeze(0), float('inf'))
        return scores


