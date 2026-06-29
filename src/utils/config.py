"""Attribute-accessible nested configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Union


class Config(Mapping):
    """A read-friendly nested config supporting both attribute and item access.

    Nested dictionaries are recursively wrapped so that
    ``cfg.model.use_adv`` and ``cfg["model"]["use_adv"]`` are equivalent.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = Config(value) if isinstance(value, dict) else value

    # -- attribute / item access -------------------------------------------
    def __getattr__(self, item: str) -> Any:
        try:
            return self._data[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = Config(value) if isinstance(value, dict) else value

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back into plain dictionaries."""
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self.to_dict()!r})"


def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML file into a :class:`Config`.

    Args:
        path: Path to a ``.yaml`` file.

    Returns:
        Parsed configuration.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)
