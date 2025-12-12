from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from src.dsl.program import Program
from src.dsl.scorer import ProgramScorer


@dataclass
class TaskEngine:
    model: torch.nn.Module
    tokenizer: object
    program: Program
    scorer: ProgramScorer

    @classmethod
    def from_spec(cls, model, tokenizer, spec, device):
        program = Program(model, tokenizer, spec, device)
        scorer = ProgramScorer(model, program)
        return cls(model=model, tokenizer=tokenizer, program=program, scorer=scorer)

    @property
    def device(self):
        return self.program.device

    @property
    def vocab_size(self) -> int:
        return self.program.vocab_size

    @property
    def variable_total_length(self) -> int:
        return self.program.total_var_len

    @torch.inference_mode()
    def compute_loss(self, variable_tokens: torch.Tensor) -> torch.Tensor:
        """Return per-sample objective in model's loss space (lower is better)."""
        return self.scorer.compute_loss(variable_tokens)

    def compute_loss_and_vocab_grads(self, variable_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (losses [B], gradient_vocab_scores [B, Lvar, V])."""
        total, grads = self.scorer.compute_loss_and_grads(variable_tokens)
        # Concatenate grads in registry order
        names = [n for n, _ in self.program.vars]
        concat = torch.cat([grads[n] for n in names], dim=1)
        
        return total, concat

    def decode_variables(self, variable_tokens: torch.Tensor) -> Dict[str, str]:
        return self.program.decode_vars(variable_tokens, self.tokenizer)


