from dataclasses import dataclass
from typing import Dict, Any

import torch

from .banned import BannedIdsConfig, get_banned_ids


@dataclass
class ConstraintPolicy:
    only_ascii: bool = False
    emoji: bool = False
    ban_params_ids: bool = True

    @classmethod
    def from_map_hyperparams(cls, hyper_params: Dict[str, Any]):
        constraints = hyper_params.get("constraints", {}) if isinstance(hyper_params, dict) else {}
        return cls(
            only_ascii=bool(constraints.get("only_ascii", False)),
            emoji=bool(constraints.get("emoji", False)),
            ban_params_ids=bool(constraints.get("ban_params_ids", True)),
        )

    def banned_token_ids(self, tokenizer, program) -> torch.Tensor:
        text_ids = program.unique_fixed_ids() if self.ban_params_ids else None
        banned = get_banned_ids(BannedIdsConfig(
            tokenizer=tokenizer,
            text_ids=text_ids,
            only_ascii=self.only_ascii,
            emoji=self.emoji,
        ))
        return banned.to(program.device)


