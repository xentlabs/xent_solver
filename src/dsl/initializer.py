import torch


def random_flat_variables(vocab_size: int, total_var_len: int, batch_size: int, banned_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    banned_ids = banned_ids.to(device=device, dtype=torch.long).unique()
    allowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if banned_ids.numel() > 0:
        allowed_mask[banned_ids] = False
    allowed_ids = torch.nonzero(allowed_mask, as_tuple=False).squeeze(-1)
    if allowed_ids.numel() == 0:
        raise ValueError("No allowed token ids available after applying banned_ids")
    rnd_idx = torch.randint(0, allowed_ids.numel(), (batch_size, total_var_len), device=device)
    return allowed_ids[rnd_idx]


