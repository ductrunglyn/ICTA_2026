"""A minimal name -> object registry used for experiments and components."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, Tuple, TypeVar

T = TypeVar("T")


class Registry:
    """Tiny registry mapping string keys to callables/classes.

    Example:
        >>> MODELS = Registry("models")
        >>> @MODELS.register("net")
        ... class Net: ...
        >>> MODELS.get("net")  # returns the Net class
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: Dict[str, object] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator registering ``obj`` under ``key``."""

        def _wrap(obj: T) -> T:
            if key in self._store:
                raise KeyError(f"'{key}' already registered in {self.name}")
            self._store[key] = obj
            return obj

        return _wrap

    def add(self, key: str, obj: object) -> None:
        """Register ``obj`` imperatively."""
        if key in self._store:
            raise KeyError(f"'{key}' already registered in {self.name}")
        self._store[key] = obj

    def get(self, key: str) -> object:
        """Retrieve a registered object, raising ``KeyError`` if missing."""
        if key not in self._store:
            raise KeyError(f"'{key}' not found in registry '{self.name}'. "
                           f"Available: {sorted(self._store)}")
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[Tuple[str, object]]:
        return iter(self._store.items())

    def __len__(self) -> int:
        return len(self._store)
