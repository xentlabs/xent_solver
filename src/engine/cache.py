import torch
from typing import Callable

class GradientCache:
    def __init__(self):
        self._flat = None
        self._total = None
        self._grads = None

    def compute(self, flat: torch.Tensor, compute_fn: Callable):
        B = flat.shape[0]
        
        # Cold start
        if self._flat is None or self._flat.numel() == 0 or self._total is None or self._grads is None:
            total, grads = compute_fn(flat)
            self._flat = flat.clone()
            self._total = total.detach().clone()
            self._grads = grads.detach().clone()
            return total, grads

        match = (flat.unsqueeze(1) == self._flat.unsqueeze(0)).all(dim=2)  # [B, B_prev]
        reuse_mask = match.any(dim=1)
        total_out = torch.empty(B, dtype=self._total.dtype, device=self._total.device)
        grads_out = torch.empty((B, *self._grads.shape[1:]), dtype=self._grads.dtype, device=self._grads.device)
        if reuse_mask.any():
            src = match[reuse_mask].float().argmax(dim=1).long()
            total_out[reuse_mask] = self._total[src]
            grads_out[reuse_mask] = self._grads[src]
        new_mask = ~reuse_mask
        if new_mask.any():
            idx = new_mask.nonzero(as_tuple=True)[0]
            tot_new, grads_new = compute_fn(flat[idx])
            total_out[new_mask] = tot_new
            grads_out[new_mask] = grads_new
        self._flat = flat.clone()
        self._total = total_out.detach().clone()
        self._grads = grads_out.detach().clone()
        return total_out, grads_out
