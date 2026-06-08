"""Tests for dataclass utils / helpers."""

from dataclasses import dataclass

import pytest

from vllm_omni.utils.dataclass_utils import trackable, trackable_to_kwargs


def test_trackable_kwargs():
    """Ensure we can track classes generically."""

    @trackable
    class NotADataClass:
        def __init__(self, foo, bar, baz=128, *args, **kwargs):
            pass

    obj = NotADataClass(foo=32, bar=64)
    # Only foo / bar were initially passed
    assert obj._init_kwargs == {"foo", "bar"}


def test_trackable_dataclass():
    """Ensure we can track classes dataclasses."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(foo=32, bar=64)
    assert obj._init_kwargs == {"foo", "bar"}


@pytest.mark.xfail(reason="positional args not yet tracked")
def test_trackable_positional_args():
    @trackable
    class NotADataClass:
        def __init__(self, foo, bar, baz=128, *args, **kwargs):
            pass

    obj = NotADataClass(32, 64)
    # Even though they were positional, this should be fine also
    assert obj._init_kwargs == {"foo", "bar"}


def test_trackable_to_kwargs():
    """Ensure a registered trackable can be filtered down to kwargs."""

    @trackable
    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(foo=32, bar=64)
    res = trackable_to_kwargs(obj)
    assert res == {"foo": 32, "bar": 64}


def test_trackable_to_kwargs_raises_with_bad_types():
    """Ensure a non trackable raises TypeError if we try to filter to kwargs."""

    @dataclass
    class MyDataClass:
        foo: int = 0
        bar: int = 0
        baz: int = 128

    obj = MyDataClass(foo=32, bar=64)
    with pytest.raises(TypeError):
        trackable_to_kwargs(obj)
