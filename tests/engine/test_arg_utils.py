"""
Tests for Omni config utils. For stability, these tests should largely be
invariant to the specific attributes of vLLM config except in cases where we
explicitly patch values that differ from vLLM.
"""

from dataclasses import asdict

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs, OmniEngineArgs


def test_default_async_engine_args_match_sync_engine_args():
    """Ensure that all keys/values in sync omni engine args are aligned
    with async omni engine args, and that any keys in the async omni engine
    args not in the sync omni args are aligned with vLLM's asyncEngineArgs.
    """
    sync_vllm_dict = asdict(EngineArgs())
    async_vllm_dict = asdict(AsyncEngineArgs())
    diff_keys = async_vllm_dict.keys() - sync_vllm_dict.keys()
    # Generally, EngineArgs keys are a subset of AsyncEngineArgs keys
    assert len(sync_vllm_dict.keys() - async_vllm_dict.keys()) == 0

    sync_omni_dict = asdict(OmniEngineArgs())
    async_omni_dict = asdict(AsyncOmniEngineArgs())
    for async_key in diff_keys:
        # Any values in not in the omni sync dataclass should
        # directly match the async vLLM async dataclass; remove
        # it & compare.
        assert async_key in async_vllm_dict
        vllm_async_val = async_vllm_dict.pop(async_key)
        omni_async_val = async_omni_dict.pop(async_key)
        assert omni_async_val == vllm_async_val

    # After removing async only keys from our AsyncOmniEngineArgs,
    # we should have the same dict as the OmniEngineArgs.
    assert async_omni_dict == sync_omni_dict


def test_sync_config_is_omni():
    """Ensure create_model_config gives the right type."""
    cfg = AsyncOmniEngineArgs().create_model_config()
    assert isinstance(cfg, OmniModelConfig)


def test_async_config_is_omni():
    """Ensure create_model_config gives the right type."""
    cfg = OmniEngineArgs().create_model_config()
    assert isinstance(cfg, OmniModelConfig)


def test_async_engine_args_mro():
    """Ensure .mro is correct to prevent issues with config creation."""
    mro = AsyncOmniEngineArgs.mro()
    omni_eng_idx = mro.index(OmniEngineArgs)
    async_args_idx = mro.index(AsyncEngineArgs)
    # > 0 since the first entry is AsyncOmniEngineArgs
    assert omni_eng_idx > 0
    assert async_args_idx > 0
    assert omni_eng_idx < async_args_idx
