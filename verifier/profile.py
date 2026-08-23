from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    version: str
    path: Path
    semantics: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    causality: dict[str, Any] = field(default_factory=dict)

    @property
    def mandatory_checks(self) -> set[str]:
        return {name for name, required in self.checks.items() if required is True}

    def get(self, name: str, default: Any = None) -> Any:
        return self.semantics.get(name, default)


def load_profile(path: str | Path) -> StrategyProfile:
    p = Path(path).resolve()
    with p.open("rb") as handle:
        raw = tomllib.load(handle)
    reserved = {"name", "version", "checks", "causality", "semantics"}
    semantics = {k: v for k, v in raw.items() if k not in reserved}
    semantics.update(raw.get("semantics", {}))
    return StrategyProfile(
        name=str(raw.get("name", p.parent.name)),
        version=str(raw.get("version", "UNVERSIONED")),
        path=p,
        semantics=semantics,
        checks={str(k): bool(v) for k, v in raw.get("checks", {}).items()},
        causality=dict(raw.get("causality", {})),
    )
