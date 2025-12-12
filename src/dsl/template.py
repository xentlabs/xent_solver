from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Union, Tuple

import torch

from src.dsl.primitives import Symbol, Part, Composition, ProgramSpec, Seq, Compositions, Ensure


Kind = Literal["fixed", "variable"]

def Fixed(text: str) -> Symbol:
    return Symbol(name="", kind="fixed", text=str(text))

def Variable(length: int) -> Symbol:
    L = int(length)
    if L <= 0:
        raise ValueError("Variable length must be positive")
    return Symbol(name="", kind="variable", length=L)

class Symbols:
    def __init__(self, specs: Union[Dict[str, Union[Symbol, str, int]], None] = None, **kwargs: Union[Symbol, str, int]):
        self.symbols: Dict[str, Symbol] = {}
        if specs is not None:
            self.add(specs)
        if kwargs:
            self.add(kwargs)

    def add(self, specs: Dict[str, Union[Symbol, str, int]]):
        for name, inst in specs.items():
            if isinstance(inst, Symbol):
                s = inst
            elif isinstance(inst, str):
                s = Fixed(text=inst)
            elif isinstance(inst, int):
                s = Variable(length=inst)
            else:
                raise ValueError(f"Invalid symbol spec for {name}: {type(inst)}")

            # Ensure the symbol’s name matches the key
            s.name = str(name)
            # Validate required fields
            if s.kind == "fixed":
                if s.text is None and s.ids is None:
                    raise ValueError(f"Fixed symbol {name} requires text or ids")
            if s.kind == "variable":
                if not isinstance(s.length, int) or s.length <= 0:
                    raise ValueError(f"Variable symbol {name} must define a positive length")
            self.symbols[name] = s

    def __getitem__(self, name: str) -> Symbol:
        return self.symbols[name]

    def as_dict(self) -> Dict[str, Symbol]:
        return self.symbols


@dataclass
class ProgramSpecTemplate:
    goal: Literal["minimize", "maximize"]
    symbols: Union[Dict[str, Symbol], Symbols] = None
    compositions: List[Union[Composition, Sequence[Composition]]] = None
    ensures: List[Union[Compositions, Ensure]] | None = None
    objective: Union[Composition, Compositions, Sequence[Composition]] | None = None

    def __post_init__(self):
        # Handle objective logic
        if self.objective is not None:
            if self.compositions is not None:
                raise ValueError("Cannot specify both 'objective' and 'compositions'")
            self.compositions = self.objective
        
        if self.compositions is None:
            raise ValueError("Must specify either 'objective' or 'compositions'")

        # Normalize symbols to a plain dict
        if isinstance(self.symbols, Symbols):
            self.symbols = self.symbols.as_dict()

        container = self.compositions if isinstance(self.compositions, (list, tuple)) else [self.compositions]
        
        flat_objective_items = []
        def _collect(items):
            if isinstance(items, Composition):
                flat_objective_items.append(items)
            elif isinstance(items, Compositions):
                flat_objective_items.extend(items.items)
            elif isinstance(items, (list, tuple)):
                for it in items:
                    _collect(it)
            else:
                raise TypeError(f"Invalid compositions entry: {type(items)}")
        
        _collect(container)
        self.compositions = Compositions(flat_objective_items)
        
        self.ensures = []
        for e in (self.ensures or []):
            if isinstance(e, Composition):
                self.ensures.append(Compositions([e]))
            elif isinstance(e, Compositions):
                self.ensures.append(e)
            elif isinstance(e, Ensure):
   
                if e.comparison.operator == ">":
                    # left > right => left - right > 0
                    expr = Compositions(e.comparison.left) - Compositions(e.comparison.right)
                elif e.comparison.operator == "<":
                    # left < right => right - left > 0
                    expr = Compositions(e.comparison.right) - Compositions(e.comparison.left)
                else:
                    raise NotImplementedError(f"Unsupported operator {e.comparison.operator}")
                self.ensures.append(expr)
            else:
                raise TypeError(f"Invalid ensure entry: {type(e)}")

        # Auto-extract symbols if not provided
        if self.symbols is None:
            self.symbols = {}
            for comp in self.compositions:
                for part in comp.parts:
                    if part.symbol:
                        self.symbols[part.name] = part.symbol
            for expr in self.ensures:
                for comp in expr:
                    for part in comp.parts:
                        if part.symbol:
                            self.symbols[part.name] = part.symbol


def materialize_spec(template: ProgramSpecTemplate, model, tokenizer, *, bos: bool) -> ProgramSpec:
    """
    Create a concrete ProgramSpec from a template using model/tokenizer.
    """
    device = model.device
    symbols: Dict[str, Symbol] = {}

    # Optional BOS
    if bos:
        if model.config.bos_token_id is None:
            raise ValueError("BOS token ID is not set in model config")
        bos_ids = torch.tensor([[model.config.bos_token_id]], dtype=torch.long, device=device)
        symbols["bos"] = Symbol("bos", kind="fixed", ids=bos_ids)

    # Materialize symbols
    for name, s in template.symbols.items():
        if s.kind == "fixed":
            if s.ids is None:
                if s.text is None:
                    raise ValueError(f"Fixed symbol {name} requires text or ids")
                ids = tokenizer(s.text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            else:
                ids = s.ids.to(device=device, dtype=torch.long)
            symbols[name] = Symbol(name, kind="fixed", ids=ids)
        else:
            length = s.length
            if not isinstance(length, int) or length <= 0:
                raise ValueError(f"Variable {name} must define a positive length")
            symbols[name] = Symbol(name, kind="variable", length=length)

    cache: Dict[Tuple, Tuple[int, Composition]] = {}
    basis: List[Composition] = []

    def get_basis_index(comp: Composition) -> int:
        parts_key = tuple((p.name, p.coeff) for p in comp.parts)
        
        if parts_key not in cache:
            mat_parts: List[Part] = []
            if "bos" in symbols:
                mat_parts.append(Part("bos", 0.0))
            mat_parts.extend(comp.parts)
            
            all_fixed = all(symbols[part.name].kind == "fixed" for part in mat_parts)
            
            c = Composition(
                parts=mat_parts,
                weight=1.0,
                constant_only=all_fixed
            )
            idx = len(basis)
            basis.append(c)
            cache[parts_key] = (idx, c)
        
        return cache[parts_key][0]

    goal_sign = -1.0 if template.goal == "maximize" else 1.0

    objective_weights = [0.0] * 1024 
    
    def add_weight(idx: int, w: float, weights_list: List[float]):
        if idx >= len(weights_list):
            weights_list.extend([0.0] * (idx - len(weights_list) + 1))
        weights_list[idx] += w

    for comp in template.compositions:
        idx = get_basis_index(comp)
        # Accumulate weight * goal_sign
        add_weight(idx, comp.weight * goal_sign, objective_weights)
    
    objective_weights = objective_weights[:len(basis)]
    if len(objective_weights) < len(basis):
        objective_weights.extend([0.0] * (len(basis) - len(objective_weights)))

    ensure_weights_list: List[List[float]] = []
    for expr in template.ensures:
        w_vec = [0.0] * len(basis)
        for comp in expr:
            idx = get_basis_index(comp)
            add_weight(idx, comp.weight, w_vec) 
        if len(w_vec) < len(basis):
            w_vec.extend([0.0] * (len(basis) - len(w_vec)))
        
        ensure_weights_list.append(w_vec)

    N = len(basis)
    if len(objective_weights) < N:
        objective_weights.extend([0.0] * (N - len(objective_weights)))
    
    for i in range(len(ensure_weights_list)):
        if len(ensure_weights_list[i]) < N:
            ensure_weights_list[i].extend([0.0] * (N - len(ensure_weights_list[i])))

    return ProgramSpec(
        symbols=symbols, 
        basis=basis, 
        objective_weights=objective_weights, 
        goal=template.goal, 
        ensure_weights=ensure_weights_list,
        compositions=basis 
    )


SymExpr = Union[Symbol, Seq]

def _names(expr: SymExpr) -> List[str]:
    if isinstance(expr, Symbol):
        return [expr.name]
    if isinstance(expr, Seq):
        return [s.name for s in expr.items]
    raise TypeError(f"Unsupported symbol expression: {type(expr)}")

def _extract_symbols(expr: SymExpr) -> List[Symbol]:
    if isinstance(expr, Symbol):
        return [expr]
    if isinstance(expr, Seq):
        return expr.items
    raise TypeError(f"Unsupported symbol expression: {type(expr)}")

def _parts(expr: SymExpr, coeff: float) -> List[Part]:
    return [Part(s.name, coeff, symbol=s) for s in _extract_symbols(expr)]

def xent(seq: SymExpr, ctx: SymExpr | None = None) -> Composition:
    """
    xent(s | ctx) = [ (ctx,0), (s,1) ]  (ctx can be Symbol or Seq)
    xent(s)       = [ (s,1) ]
    """
    ctx_parts = _parts(ctx, 0.0) if ctx is not None else []
    return Composition(parts=ctx_parts + _parts(seq, 1.0), weight=1.0)

def nex(seq: SymExpr, ctx: SymExpr | None = None) -> Composition:
    c = xent(seq, ctx)
    return Composition(parts=c.parts, weight=-1.0)

def xed(seq: SymExpr, ctx: SymExpr) -> Compositions:
    return Compositions([xent(seq), nex(seq, ctx)])

def dex(seq: SymExpr, ctx: SymExpr) -> Compositions:
    return Compositions([nex(seq), xent(seq, ctx)])
