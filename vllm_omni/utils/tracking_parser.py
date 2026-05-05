# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from typing import Any

from vllm.utils.argparse_utils import FlexibleArgumentParser

UNSET = object()


class TrackingNamespace(argparse.Namespace):
    """Proxy the wraps an argparse namespace with explicit keys, which can
    can be filtered down to a dict containing only explicitly passed values.
    """

    def __init__(self, unfiltered_ns: argparse.Namespace, explicit_keys: frozenset[str]) -> None:
        self.unfiltered_ns = unfiltered_ns
        self.explicit_keys = explicit_keys

    def get_explicit_kwargs_dict(self):
        """Given an instance of this class, return a dict with all dropped items."""
        return {k: v for k, v in vars(self.unfiltered_ns).items() if k in self.explicit_keys}

    def __getattr__(self, name: str) -> Any:
        # Any attribute access is forwarded to the real argument group.
        return getattr(self.unfiltered_ns, name)


class TrackingGroup:
    """Proxy that wraps an argument group and its corresponding shadow group."""

    def __init__(
        self,
        real_group: argparse._ArgumentGroup,
        shadow_group: argparse._ArgumentGroup,
    ):
        self._real = real_group
        self._shadow = shadow_group

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add an argument to the real group and to the shadow group."""
        action = self._real.add_argument(*args, **kwargs)
        default_kwargs = {**kwargs, "default": UNSET}
        self._shadow.add_argument(*args, **default_kwargs)
        return action

    def __getattr__(self, name: str) -> Any:
        # Any attribute access is forwarded to the real argument group.
        return getattr(self._real, name)


class TrackingSubparsers:
    """Proxy that wraps a subparser and its corresponding shadow subparser."""

    def __init__(
        self,
        real_sub: argparse._SubParsersAction,
        shadow_sub: argparse._SubParsersAction,
    ):
        self._real = real_sub
        self._shadow = shadow_sub

    def add_parser(self, name, *args, **kwargs):
        """Add a parser to the encapsulated real parser and its shadow."""
        real_parser = self._real.add_parser(name, *args, **kwargs)
        # real_parser is a TrackingArgumentParser with its own _shadow.
        # Reuse that shadow as the parent shadow's child — so when
        # real_parser.add_argument() mirrors to real_parser._shadow,
        # the parent's shadow sees it too.
        self._shadow._name_parser_map[name] = real_parser._shadow
        return real_parser

    def __getattr__(self, name: str) -> Any:
        # Any attribute access is forwarded to the real subparser.
        return getattr(self._real, name)


class TrackingArgumentParser(FlexibleArgumentParser):
    """Drop-in replacement for FlexibleArgumentParser, which tracks keys that
    were explicitly passed as args.

    Unfortunately, Argparse does not provide an easy way of doing this without
    depending on a lot of internal attributes, so we implement it by instead
    using a 'shadow' parser, which is essentially a clone of the parser, where
    defaults are overridden to `None`. By comparing the parser against its
    shadow, we can tell which values were passed in a non-destructive manner.
    """

    def __init__(self, *args, **kwargs):
        # NOTE: We have to define the shadow parser before calling init,
        # with add_help=False, since otherwise init will call add_argument
        # and delegate to the override on this class and cause problems.
        shadow_kwargs = {**kwargs, "add_help": False}
        self._shadow = FlexibleArgumentParser(*args, **shadow_kwargs)
        self._explicit_keys: frozenset[str] = frozenset()
        super().__init__(*args, **kwargs)

    @property
    def explicit_keys(self) -> frozenset[str]:
        """The set of keys that were passed explicitly."""
        return self._explicit_keys

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add an arg to the parser & the shadow, where the latter has None for the default."""
        action = super().add_argument(*args, **kwargs)
        shadow_kwargs = {**kwargs, "default": UNSET}
        self._shadow.add_argument(*args, **shadow_kwargs)
        return action

    def add_argument_group(self, *args, **kwargs) -> TrackingGroup:
        real_group = super().add_argument_group(*args, **kwargs)
        shadow_group = self._shadow.add_argument_group(*args, **kwargs)
        return TrackingGroup(real_group, shadow_group)

    def add_mutually_exclusive_group(self, *args, **kwargs) -> TrackingGroup:
        real_group = super().add_mutually_exclusive_group(*args, **kwargs)
        shadow_group: argparse._MutuallyExclusiveGroup = self._shadow.add_mutually_exclusive_group(*args, **kwargs)
        return TrackingGroup(real_group, shadow_group)

    def add_subparsers(self, *args, **kwargs) -> TrackingSubparsers:
        real_sub = super().add_subparsers(*args, **kwargs)
        shadow_sub = self._shadow.add_subparsers(*args, **kwargs)
        return TrackingSubparsers(real_sub, shadow_sub)

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> TrackingNamespace:
        """Parse the args on the real/shadow parser & set the frozen explicit keys."""
        # Only the real parser should use the namespace if one is,
        # given since shadow parser will set its own defaults to None.
        real_ns = super().parse_args(args, namespace)
        shadow_ns = self._shadow.parse_args(args)
        # Explicit keys are entries in the shadow namespace that aren't UNSET.
        # NOTE: This is distinct from `None`, since there are cases where `None`
        # user can pass explicit values that set `None` on the namespace
        self._explicit_keys = frozenset(k for k, v in vars(shadow_ns).items() if v is not UNSET)
        return TrackingNamespace(real_ns, self._explicit_keys)

    def parse_known_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        """Parse the knwown args on the real/shadow parser & set the frozen explicit keys."""
        real_ns, remaining = super().parse_known_args(args, namespace)
        shadow_ns, _ = self._shadow.parse_known_args(args)
        self._explicit_keys = frozenset(k for k, v in vars(shadow_ns).items() if v is not UNSET)
        return real_ns, remaining
