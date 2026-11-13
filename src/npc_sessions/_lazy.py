"""Deferred imports of heavy third-party modules.

Usage (keeps static type checkers and annotations unaffected)::

    if typing.TYPE_CHECKING:
        import h5py
    else:
        h5py = lazy_import("h5py")

The real module (and any listed submodules) is imported on first attribute access.
"""

from __future__ import annotations

import functools
import importlib
import threading
import types
from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

from typing_extensions import Self

_lock = threading.RLock()


class _LazyModule(types.ModuleType):
    def __init__(self, name: str, submodules: tuple[str, ...]) -> None:
        super().__init__(name)
        self.__dict__["_lazy_submodules"] = submodules
        self.__dict__["_lazy_real"] = None

    def _lazy_load(self) -> types.ModuleType:
        real = self.__dict__["_lazy_real"]
        if real is None:
            with _lock:
                real = importlib.import_module(self.__name__)
                for sub in self.__dict__["_lazy_submodules"]:
                    importlib.import_module(sub)
                self.__dict__["_lazy_real"] = real
        return real

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            if attr in ("__path__", "__file__", "__spec__", "__wrapped__"):
                pass
            else:
                raise AttributeError(attr)
        return getattr(self._lazy_load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._lazy_load(), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._lazy_load(), attr)

    def __dir__(self) -> list[str]:
        return dir(self._lazy_load())

    def __repr__(self) -> str:
        return f"<lazy module {self.__name__!r}>"


def lazy_import(name: str, *submodules: str) -> Any:
    """Return a stand-in for top-level module `name`, imported on first use."""
    return _LazyModule(name, submodules)


def lazy_njit(*njit_args: Any, **njit_kwargs: Any) -> Any:
    """`numba.njit(**kwargs)` that defers importing numba and compiling until first call."""

    def decorator(func: Any) -> Any:
        compiled: list[Any] = []

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not compiled:
                with _lock:
                    if not compiled:
                        import numba

                        compiled.append(numba.njit(*njit_args, **njit_kwargs)(func))
            return compiled[0](*args, **kwargs)

        wrapper.py_func = func  # type: ignore[attr-defined]
        return wrapper

    return decorator


class lazy_class:
    """Class attribute whose value (a class) is built by `factory` on first access.

    For nested classes that subclass heavy third-party classes.
    """

    def __init__(self, factory: Callable[[], type]) -> None:
        self._factory = factory
        self._cls: type | None = None
        self._owner: type | None = None
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._owner, self._name = owner, name

    def __get__(self, instance: object, owner: type | None = None) -> type:
        if self._cls is None:
            with _lock:
                if self._cls is None:
                    cls = self._factory()
                    if self._owner is not None and self._name is not None:
                        cls.__qualname__ = f"{self._owner.__qualname__}.{self._name}"
                        cls.__module__ = self._owner.__module__
                    self._cls = cls
        return self._cls


_T = TypeVar("_T")
_NOT_FOUND = object()


class cached_property(functools.cached_property, Generic[_T]):  # type: ignore[type-arg]
    """Same as `npc_io.cached_property` (stdlib `cached_property` without the
    instance lock), duplicated here so that importing it doesn't import `npc_io`
    (and boto3 etc.)."""

    func: Callable[[Any], _T]

    def __init__(self, func: Callable[[Any], _T]) -> None:
        self.__module__ = func.__module__  # allows doctests to run
        super().__init__(func)

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> _T: ...

    def __get__(self, instance, owner=None) -> Self | _T | Any:
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val
