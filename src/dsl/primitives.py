from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Literal, Union, Tuple, Iterable

import torch

Kind = Literal["fixed", "variable"]


@dataclass
class Symbol:
    name: str
    kind: Kind
    text: Optional[str] = None           # for fixed templates; materialized into ids
    ids: Optional[torch.Tensor] = None   # for fixed: [1, L] int64 on device
    length: Optional[int] = None         # for variable

    # Concatenation produces a Seq wrapper (not a Python list)
    def __add__(self, other: Union["Symbol", "Seq"]) -> "Seq":
        if isinstance(other, Seq):
            return Seq([self] + other.items)
        if isinstance(other, Symbol):
            return Seq([self, other])
        raise TypeError(f"Cannot add Symbol and {type(other)}")

    def __radd__(self, other: "Seq") -> "Seq":
        if isinstance(other, Seq):
            return Seq(other.items + [self])
        raise TypeError(f"Cannot add {type(other)} and Symbol")


@dataclass
class Seq:
    items: List[Symbol]

    def __add__(self, other: Union[Symbol, "Seq"]) -> "Seq":
        if isinstance(other, Seq):
            return Seq(self.items + other.items)
        if isinstance(other, Symbol):
            return Seq(self.items + [other])
        raise TypeError(f"Cannot add Seq and {type(other)}")

    def __iter__(self):
        return iter(self.items)


@dataclass
class Fixed(Symbol):
    ids: Optional[torch.Tensor] = None
    length: Optional[int] = None

@dataclass
class Variable(Symbol):
    ids: Optional[torch.Tensor] = None
    length: Optional[int] = None

@dataclass
class Part:
    name: str
    coeff: float
    symbol: Optional["Symbol"] = field(default=None, repr=False, compare=False)

@dataclass(eq=False)
class Composition:
    parts: List[Part]
    weight: float = 1.0
    constant_only: bool = False
    __hash__ = object.__hash__

    def _comparison(self, other, operator: str):
        if operator == ">": return self - other
        if operator == "<": return other - self
        raise NotImplementedError(f"Ensure only supports < and > operators, got {operator}")

    def __lt__(self, other: "Composition"):
        return self._comparison(other, "<")

    def __gt__(self, other: "Composition"):
        return self._comparison(other, ">")

    def __mul__(self, other: float) -> "Composition":
        if isinstance(other, (int, float)):
            return replace(self, weight=self.weight * float(other))
        return NotImplemented
    
    def __rmul__(self, other: float) -> "Composition":
        return self.__mul__(other)
    
    def __neg__(self) -> "Composition":
        return self * -1.0
    
    def __add__(self, other) -> "Compositions":
        if isinstance(other, Composition):
            return Compositions([self, other])
        if isinstance(other, Compositions):
            return Compositions([self] + list(other.items))
        return NotImplemented

    def __sub__(self, other) -> "Compositions":
        return self + (-other)


class Compositions:
    def __init__(self, items: Iterable[Composition]):
        normalized = tuple(items)
        if not normalized:
            raise ValueError("Compositions cannot be empty")
        if not all(isinstance(entry, Composition) for entry in normalized):
            raise TypeError("Compositions must contain Composition objects")
        self.items: Tuple[Composition, ...] = normalized

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def _comparison(self, other, operator: str):
        if operator == ">": return self - other
        if operator == "<": return other - self
        raise NotImplementedError(f"Ensure only supports < and > operators, got {operator}")

    def __lt__(self, other):
        return self._comparison(other, "<")

    def __gt__(self, other):
        return self._comparison(other, ">")

    def __mul__(self, other: float) -> "Compositions":
        if isinstance(other, (int, float)):
            return Compositions([c * other for c in self.items])
        return NotImplemented
    
    def __rmul__(self, other: float) -> "Compositions":
        return self.__mul__(other)
    
    def __neg__(self) -> "Compositions":
        return self * -1.0
    
    def __add__(self, other) -> "Compositions":
        if isinstance(other, Composition):
            return Compositions(list(self.items) + [other])
        if isinstance(other, Compositions):
            return Compositions(list(self.items) + list(other.items))
        return NotImplemented
    
    def __sub__(self, other) -> "Compositions":
        return self + (-other)


def _flatten_group(value) -> Tuple[Composition, ...]:
    if isinstance(value, Composition):
        return (value,)
    if isinstance(value, Compositions):
        return value.items
    if isinstance(value, (list, tuple)):
        out: List[Composition] = []
        for it in value:
            out.extend(_flatten_group(it))
        if not out:
            raise ValueError("Ensure comparison set cannot be empty")
        return tuple(out)
    raise TypeError("Expected Composition or (nested) list/tuple of Composition")


def _to_group(value) -> Tuple[Composition, ...]:
    return _flatten_group(value)


@dataclass(frozen=True)
class LossComparison:
    left: Tuple[Composition, ...]
    right: Tuple[Composition, ...]
    operator: Literal["<", "<=", ">", ">=", "==", "!="]


@dataclass
class Ensure:
    comparison: LossComparison
    scale: float = 4.0
    target: Optional[Tuple[Composition, ...]] = None
    apply_when_not_met: bool = True

    def __post_init__(self):
        left = self.comparison.left
        right = self.comparison.right
        target = _to_group(self.target if self.target is not None else left)
        if not any(not comp.constant_only for comp in target):
            raise ValueError("Ensure target must include at least one variable composition")
        object.__setattr__(self, "comparison", LossComparison(left, right, self.comparison.operator))
        object.__setattr__(self, "target", target)


@dataclass
class ProgramSpec:
    symbols: Dict[str, Symbol]
    basis: List[Composition]
    objective_weights: List[float]
    goal: Literal["minimize", "maximize"]
    ensure_weights: List[List[float]] = field(default_factory=list)
    compositions: List[Composition] = field(default_factory=list) 
    ensures: List[Ensure] = field(default_factory=list)
