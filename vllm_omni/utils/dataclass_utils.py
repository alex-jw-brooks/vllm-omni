# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

_T = TypeVar("_T")


def trackable(cls: type[_T]) -> type[_T]:
    """Decorator that wraps __init__ to track which args/kwargs were explicitly
    passed by the caller without special handling. This is useful for a variety
    of situations, e.g., merge a user's passed sampling params with the default
    values provided by a pipeline.

    NOTE: This decorator preserves the original __init__ signature for
    type checkers while adding runtime tracking of explicitly-passed kwargs.
    """
    original_init: Callable[..., None] = cls.__init__
    sig = inspect.signature(original_init)

    @wraps(original_init)
    def new_init(self: _T, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Map passed args/kwargs to wrapped initializer.
        bound = sig.bind(self, *args, **kwargs)
        bound.arguments.pop("self", None)
        self._init_kwargs: set[str] = set(bound.arguments)  # type: ignore[attr-defined]

    # Replace __init__ - type: ignore needed due to limitations in typing dynamic method replacement
    cls.__init__ = new_init  # type: ignore[method-assign]
    return cls


def trackable_to_kwargs(obj):
    if not hasattr(obj, "_init_kwargs"):
        raise TypeError(f"Provided object of type {type(obj)} is not registered as trackable")
    return {kwarg: getattr(obj, kwarg) for kwarg in obj._init_kwargs}
