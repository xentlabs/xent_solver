from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .program import Program


class ProgramScorer:
    def __init__(self, model: torch.nn.Module, program: Program):
        self.model, self.program = model, program
        self.EW = model.get_input_embeddings().weight  # [V, D]
        self.constant_total_loss = torch.tensor(0.0, dtype=torch.float32, device=program.device)
        
        self.constant_comp_values: Dict[int, torch.Tensor] = {} # Basis Index -> 
        
        with torch.inference_mode():
            for idx, comp in enumerate(program.spec.basis):
                if not comp.constant_only:
                    continue
                ids, emb, w = program.render_full(idx, {}, {})
                logits = model(inputs_embeds=emb).logits
                raw = self._weighted_ce(logits, ids, w)
                val = raw.squeeze().detach()
                self.constant_comp_values[idx] = val
                
                if idx < len(program.spec.objective_weights):
                    self.constant_total_loss += val * program.spec.objective_weights[idx]

    def _weighted_ce(self, logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        pred = logits[:, :-1, :]
        tgt = ids[:, 1:]
        w = weights[:, 1:]
        ce = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            tgt.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).to(torch.float32).reshape(pred.shape[0], pred.shape[1])
        return (ce * w).sum(dim=1)

    def _compute(self, flat_var_ids: torch.Tensor, need_grads: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        full_ids, inputs_emb, weights, var_embeds = self.program.build_batch(flat_var_ids, require_grads=need_grads)

        logits = self.model(inputs_embeds=inputs_emb).logits  # [Cv*B, Lmax, V]
        
        # Compute weighted cross-entropy using the helper method
        per_sample = self._weighted_ce(logits, full_ids, weights)
        
        Cv = self.program.num_variable_compositions
        totals_by_var_row = per_sample.view(Cv, -1) # [Cv, B]
        
        row_to_basis = {r: b for b, r in self.program.variable_rows.items()}
        
        
        obj_w = torch.tensor(
            [self.program.spec.objective_weights[row_to_basis[j]] for j in range(Cv)],
            dtype=totals_by_var_row.dtype,
            device=totals_by_var_row.device
        ).unsqueeze(1) # [Cv, 1]
        
        total = (totals_by_var_row * obj_w).sum(dim=0)
        total = total + self.constant_total_loss

        
        if self.program.spec.ensure_weights:
            for w_vec_list in self.program.spec.ensure_weights:
                val_const = sum(self.constant_comp_values[i] * w_vec_list[i] for i in self.constant_comp_values if w_vec_list[i] != 0)
                val_const = torch.tensor(val_const, device=total.device, dtype=total.dtype)
                
                row_w = torch.tensor(
                    [w_vec_list[row_to_basis[j]] for j in range(Cv)],
                    dtype=totals_by_var_row.dtype,
                    device=totals_by_var_row.device
                ).unsqueeze(1)
                
                val_var = (totals_by_var_row * row_w).sum(dim=0)
                val = val_const + val_var
                
                penalty = F.relu(-val) * 4.0
                total = total + penalty

        if not need_grads:
            return total, {}

        scalar = total.mean()
        grads = torch.autograd.grad(scalar, list(var_embeds.values()), retain_graph=False, create_graph=False)
        grads_vocab: Dict[str, torch.Tensor] = {}
        for (name, emb), g in zip(var_embeds.items(), grads):
            gv = g.reshape(-1, g.shape[-1]) @ self.EW.t()
            grads_vocab[name] = gv.view(g.shape[0], g.shape[1], -1)
        return total, grads_vocab

    @torch.inference_mode()
    def compute_loss(self, flat_var_ids: torch.Tensor) -> torch.Tensor:
        total, _ = self._compute(flat_var_ids, need_grads=False)
        return total

    def compute_loss_and_grads(self, flat_var_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._compute(flat_var_ids, need_grads=True)

    def forward(self, flat_var_ids: torch.Tensor, need_grads: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._compute(flat_var_ids, need_grads)
