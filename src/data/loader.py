from dataclasses import dataclass
from typing import Any, Dict, Iterable
from pathlib import Path
import json


@dataclass
class TaskSpec:
    name: str
    details: Dict[str, Any]
    hyper_params: Dict[str, Any]


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

def load_maps(path: Path | str) -> Iterable[Dict[str, Any]]:
    """Load map specifications from JSON or Python sources."""

    path = Path(path)
    suff = path.suffix.lower()
    if suff == ".json":
        return _load_json(path)
    if suff == ".py":
        import importlib.util
        import types

        spec = importlib.util.spec_from_file_location("user_maps", str(path))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import python data file: {path}")
        module = types.ModuleType(spec.name)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        candidate = module.__dict__.get("data")
        if candidate is None:
            raise ValueError("Python data file must export `data`")
        if callable(candidate):
            candidate = candidate()
        return candidate
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


class MapDataset:
    """Dataset-like wrapper over maps for Fabric dataloader usage (batch_size=1)."""

    def __init__(self, data_path: str):
        self.path = Path(data_path)
        self._maps = list(load_maps(self.path))

    def __len__(self) -> int:
        return len(self._maps)

    def __getitem__(self, idx: int) -> TaskSpec:
        mp = self._maps[idx]
        name = mp.get("name", f"Map {idx}")
        details = mp["details"]
        hyper = mp["hyper_params"]
        return TaskSpec(name=name, details=details, hyper_params=hyper)


