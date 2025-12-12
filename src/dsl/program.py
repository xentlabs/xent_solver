from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
from .primitives import ProgramSpec, Symbol, Composition


class Program:
    def __init__(self, model, tokenizer, spec: ProgramSpec, device: torch.device):
        self.model, self.tokenizer, self.spec, self.device = model, tokenizer, spec, device
        self.embeddings = model.get_input_embeddings()
        self.vocab_size, self.embed_dim = self.embeddings.weight.shape

        # Cache fixed ids/embeds
        self.fixed_ids: Dict[str, torch.Tensor] = {}
        self.fixed_embeds: Dict[str, torch.Tensor] = {}
        for sym in spec.symbols.values():
            if sym.kind == "fixed":
                assert sym.ids is not None, f"Fixed symbol {sym.name} requires ids"
                ids = sym.ids.to(device=device, dtype=torch.long)
                self.fixed_ids[sym.name] = ids
                self.fixed_embeds[sym.name] = self.embeddings(ids)

        # Variable registry and flatten layout
        self.vars: List[Tuple[str, int]] = [(s.name, s.length) for s in spec.symbols.values() if s.kind == "variable"]
        self.var_slices: Dict[str, slice] = {}
        off = 0
        for name, L in self.vars:
            assert isinstance(L, int) and L > 0, f"Variable {name} must define a positive length"
            self.var_slices[name] = slice(off, off + L)
            off += L
        self.total_var_len = off

        # Mask compositions that contain any variable parts
        self.variable_comp_mask = torch.tensor(
            [not comp.constant_only for comp in spec.basis],
            dtype=torch.bool,
            device=device,
        )
        self.num_variable_compositions = int(self.variable_comp_mask.sum().item())
        self.variable_rows: Dict[int, int] = {} # Map BASIS idx to row idx
        
        row_idx = 0
        for i, keep in enumerate(self.variable_comp_mask):
            if keep:
                self.variable_rows[i] = row_idx
                row_idx += 1

    @property
    def goal_sign(self) -> float:
        return -1.0 if self.spec.goal == "maximize" else 1.0

    def report_score(self, value) -> float:
        if isinstance(value, torch.Tensor):
            val = float(value.item())
        else:
            val = float(value)
        return -val if self.spec.goal == "maximize" else val

    def unflatten_vars(self, flat_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: flat_ids[:, slc] for name, slc in self.var_slices.items()}

    def flatten_vars(self, var_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([var_dict[name] for name, _ in self.vars], dim=1)

    def _render_composition(self, comp: Composition, var_ids: Dict[str, torch.Tensor], var_embeds: Dict[str, torch.Tensor]):
        # Concatenate parts in order (repeats allowed with different coeffs)
        ids_chunks, emb_chunks, w_chunks = [], [], []
        B = next(iter(var_ids.values())).shape[0] if var_ids else 1
        for part in comp.parts:
            sym: Symbol = self.spec.symbols[part.name]
            coeff = float(part.coeff)
            if sym.kind == "fixed":
                ids = self.fixed_ids[part.name].expand(B, -1)
                emb = self.fixed_embeds[part.name].expand(B, -1, -1)
            else:
                ids = var_ids[part.name]
                emb = var_embeds[part.name]
            ids_chunks.append(ids)
            emb_chunks.append(emb)
            if coeff == 0.0:
                w_chunks.append(torch.zeros((ids.shape[0], ids.shape[1]), dtype=torch.float32, device=ids.device))
            else:
                w_chunks.append(torch.full((ids.shape[0], ids.shape[1]), coeff, dtype=torch.float32, device=ids.device))
        ids_cat = torch.cat(ids_chunks, dim=1)
        emb_cat = torch.cat(emb_chunks, dim=1)
        w_cat = torch.cat(w_chunks, dim=1)
        return ids_cat, emb_cat, w_cat

    def render_full(self, idx: int, var_ids: Dict[str, torch.Tensor], var_embeds: Dict[str, torch.Tensor]):
        comp = self.spec.basis[idx]
        return self._render_composition(comp, var_ids, var_embeds)

    def build_batch(self, flat_var_ids: torch.Tensor, require_grads: bool = True):
        """
        Args:
            flat_var_ids: [B, total_var_len] variable token ids

        Returns:
            full_ids:   [Cv*B, Lmax]
            inputs_emb: [Cv*B, Lmax, D]
            weights:    [Cv*B, Lmax]
            var_embeds: Dict[str, Tensor] with shape [B, Lvar, D] and requires_grad=True
        """
        var_ids = self.unflatten_vars(flat_var_ids.to(self.device))
        var_embeds = {k: self.embeddings(v).detach().requires_grad_(require_grads) for k, v in var_ids.items()}

        if self.num_variable_compositions == 0:
            # No variable compositions: return empty batch tensors
            dummy = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return dummy, dummy.unsqueeze(-1), dummy, var_embeds

        comp_ids, comp_embs, comp_ws, lens = [], [], [], []
        for idx, comp in enumerate(self.spec.basis):
            if not self.variable_comp_mask[idx]:
                continue
            ids, emb, w = self._render_composition(comp, var_ids, var_embeds)
            comp_ids.append(ids)
            comp_embs.append(emb)
            comp_ws.append(w)
            lens.append(ids.shape[1])

        Lmax = max(lens)
        padded_ids, padded_embs, padded_ws = [], [], []
        for ids, emb, w in zip(comp_ids, comp_embs, comp_ws):
            pad = Lmax - ids.shape[1]
            if pad > 0:
                ids = F.pad(ids, (0, pad), value=-100)
                emb = F.pad(emb, (0, 0, 0, pad), value=0.0)
                w = F.pad(w, (0, pad), value=0.0)
            padded_ids.append(ids)
            padded_embs.append(emb)
            padded_ws.append(w)

        full_ids = torch.cat(padded_ids, dim=0)
        inputs_emb = torch.cat(padded_embs, dim=0)
        weights = torch.cat(padded_ws, dim=0)
        return full_ids, inputs_emb, weights, var_embeds

    def unique_fixed_ids(self) -> torch.Tensor:
        if not self.fixed_ids:
            return torch.empty(0, dtype=torch.long, device=self.device)
        # Each ids tensor has shape [1, L]
        cat = torch.cat([ids.to(self.device) for ids in self.fixed_ids.values()], dim=1)
        uniq = torch.unique(cat.view(-1))
        return uniq.to(dtype=torch.long, device=self.device)


    def decode_vars(self, flat_var_ids: torch.Tensor, tokenizer=None) -> Dict[str, str]:
        tok = tokenizer or self.tokenizer
        out: Dict[str, str] = {}
        
        for name, _ in self.vars:
            slc = self.var_slices[name]
            ids = flat_var_ids[0, slc]
            out[name] = tok.decode(ids.detach().cpu().tolist()) if tok is not None else str(ids.detach().cpu().tolist())
        return out
