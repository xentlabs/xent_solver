from __future__ import annotations

from typing import Tuple

import torch
from lightning.fabric import Fabric


class FabricPlus(Fabric):
    """Drop-in Fabric extension providing helper utilities used in search."""

    def global_sync_best(self, tokens: torch.Tensor, loss: float) -> Tuple[torch.Tensor, float]:
        if self.world_size == 1:
            return tokens, loss

        loss_tensor = torch.tensor([loss], device=tokens.device, dtype=torch.float32)
        gathered_losses = self.all_gather(loss_tensor).view(-1)
        gathered_tokens = self.all_gather(tokens).reshape(-1, tokens.shape[-1])
        best_idx = torch.argmin(gathered_losses)
        return gathered_tokens[best_idx].unsqueeze(0), float(gathered_losses[best_idx].item())

    def all_gather_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        
        gathered = self.all_gather(tensor)
        return gathered.reshape(-1, *tensor.shape[1:])

    def sync_best_from_pool(self, tokens: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, float]:
        local_val, local_idx = torch.min(scores, dim=0)
        local_best = tokens[local_idx].unsqueeze(0)
        return self.global_sync_best(local_best, float(local_val.detach()))


