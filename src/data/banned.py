import torch
from dataclasses import dataclass


@dataclass
class BannedIdsConfig:
    tokenizer: object
    text_ids: torch.Tensor | None = None
    only_ascii: bool = False
    emoji: bool = False


def get_banned_ids(config: BannedIdsConfig):
    out_dtype = torch.long if config.text_ids is None else config.text_ids.dtype
    banned_set = set()

    # Prohibit ids present in text (ignore -100)
    if config.text_ids is not None:
        unique_ids = torch.unique(config.text_ids)
        banned_set.update(int(x) for x in unique_ids.tolist() if int(x) >= 0)

    # ASCII-only
    if config.only_ascii:
        special_ids = set(config.tokenizer.all_special_ids)
        banned_set.update(special_ids)
        vocab_size = config.tokenizer.vocab_size
        for idx in range(vocab_size):
            if idx in special_ids:
                continue
            try:
                text = config.tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)  # type: ignore
            except Exception:
                banned_set.add(idx)
                continue
            non_ascii = [c for c in text if ord(c) >= 128]
            if not non_ascii:
                continue
            if config.emoji:
                def _is_emoji_char(ch: str) -> bool:
                    code = ord(ch)
                    return (
                        0x1F300 <= code <= 0x1F5FF or 0x1F600 <= code <= 0x1F64F or 0x1F680 <= code <= 0x1F6FF or
                        0x1F700 <= code <= 0x1F77F or 0x1F780 <= code <= 0x1F7FF or 0x1F800 <= code <= 0x1F8FF or
                        0x1F900 <= code <= 0x1F9FF or 0x1FA00 <= code <= 0x1FAFF or 0x2600 <= code <= 0x26FF or
                        0x2700 <= code <= 0x27BF or 0xFE00 <= code <= 0xFE0F or 0x1F1E6 <= code <= 0x1F1FF or
                        0x1F3FB <= code <= 0x1F3FF or code == 0x200D
                    )
                if all(_is_emoji_char(c) for c in non_ascii):
                    continue
            banned_set.add(idx)

    if len(banned_set) == 0:
        return torch.empty(0, dtype=out_dtype)
    return torch.tensor(sorted(banned_set), dtype=out_dtype)


