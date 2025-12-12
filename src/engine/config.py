from dataclasses import dataclass, field
from typing import Literal
from src.strategies.base import SelectionParams


@dataclass
class AnnealingParams:
    initial_temp: float
    min_temp: float
    decay: float
    schedule: Literal["candidate", "grad", "step"]


@dataclass
class OptimizerConfig:
    candidate_patience: int
    grad_patience: int
    annealing: AnnealingParams | None = None
    selection: SelectionParams = field(default_factory=SelectionParams)

    @classmethod
    def from_settings(cls, settings: dict) -> "OptimizerConfig":
        annealing = settings.get("annealing")
        selection = settings.get("selection", {})
        return cls(
            candidate_patience=int(settings["candidate_expl"]),
            grad_patience=int(settings["grad_expl"]),
            annealing=AnnealingParams(**annealing) if annealing else None,
            selection=SelectionParams(**selection),
        )