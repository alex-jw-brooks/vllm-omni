import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

"""
Things to test:

1. Merging for hidden states
2. Merging for multimodal
    -> Happy path
    -> No hit cache (different stage)
    -> Invalid key path (stage has wrong keys)
3. End to end?
4. CPU offload
    -> It's all on the CPU
    -> It's all on the GPU
    -> It's a mix of on the CPU and GPU
5.
"""
NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DEVICE = torch.device("cuda")
DTYPE = torch.float32
DEFAULT_SHAPE = torch.Size([NUM_BLOCKS, BLOCK_SIZE, HIDDEN_SIZE])


def build_cache_with_mm_keys(mm_cache_keys) -> OmniTensorPrefixCache:
    return OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        dtype=DTYPE,
        mm_cache_keys=mm_cache_keys,
    )


### Tests for initialization
def test_initialization_from_list_of_cache_keys():
    """Ensure that hidden states / mm outputs cache are created with the
    correct sizes by default.
    """
    mm_cache_keys = ["foo", "bar"]
    cache = build_cache_with_mm_keys(mm_cache_keys)
    assert isinstance(cache.hidden_states_cache, torch.Tensor)
    assert cache.hidden_states_cache.shape == DEFAULT_SHAPE
    assert set(mm_cache_keys) == set(cache.mm_outputs_cache.keys())
    for val in cache.mm_outputs_cache.values():
        assert isinstance(val, torch.Tensor)
        assert val.shape == DEFAULT_SHAPE


def test_initialization_from_dict_of_cache_keys():
    """Ensure that keys in the mm outputs cache can have their own feature
    sizes and fall back to the hidden states cache size if they map to None.
    """
    mm_cache_keys = {
        "foo": 100,
        "bar": 50,
        "baz": None,
    }
    cache = build_cache_with_mm_keys(mm_cache_keys)
    assert isinstance(cache.hidden_states_cache, torch.Tensor)
    assert cache.hidden_states_cache.shape == DEFAULT_SHAPE
    assert set(mm_cache_keys) == set(cache.mm_outputs_cache.keys())

    for key, val in cache.mm_outputs_cache.items():
        assert isinstance(val, torch.Tensor)
        hs_override = mm_cache_keys[key] if mm_cache_keys[key] is not None else HIDDEN_SIZE
        expected_shape = torch.Size([NUM_BLOCKS, BLOCK_SIZE, hs_override])
        assert val.shape == expected_shape


### Tests for Update
def test_update_no_multimodal():
    """Test that slot mappings act as row indices hidden states."""
    cache = build_cache_with_mm_keys(mm_cache_keys=None)

    num_tokens_unpadded = 8
    # Map the hidden states to valid & unique slots
    slot_offset = 6  # We'll put our states in slots 6, 7, 8, ..., 13
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    new_hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE, device=DEVICE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=new_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
    )

    # Ensure that if we reshape our 3D cache back to 2D, we can use the
    # indices in our slot mappings to access the hidden states as expected
    hs_rows = cache.hidden_states_cache.view(NUM_BLOCKS * BLOCK_SIZE, HIDDEN_SIZE)
    for slot_idx, new_states in zip(slot_mapping, new_hidden_states):
        slot_states = hs_rows[slot_idx]
        assert torch.all(slot_states == new_states)


@pytest.mark.parametrize(
    "mm_cache_keys",
    [
        ("foo", "bar"),  # All same feature dim (HIDDEN_SIZE)
        {"foo": 100, "bar": 50, "baz": None},  # different feature dims
    ],
)
def test_update_with_multimodal_outputs(mm_cache_keys):
    """Test that slot mappings are correct for multimodal tensors."""
    cache = build_cache_with_mm_keys(mm_cache_keys)

    num_tokens_unpadded = 8
    # Map the hidden states to valid & unique slots
    slot_offset = 6  # We'll put our states in slots 6, 7, 8, ..., 13
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    mm_outputs = {
        key: torch.rand((num_tokens_unpadded, feature_dims[key]), dtype=DTYPE, device=DEVICE) for key in mm_cache_keys
    }
    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=mm_outputs,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
    )

    for mm_key in mm_cache_keys:
        assert mm_key in cache.mm_outputs_cache
        key_feat_dim = feature_dims[mm_key]
        mm_state_rows = cache.mm_outputs_cache[mm_key].view(NUM_BLOCKS * BLOCK_SIZE, key_feat_dim)

        # Similar to hidden states, but for each key in the dict;
        # Different tensors may have different feature dims
        new_mm_states = mm_outputs[mm_key]
        for slot_idx, new_states in zip(slot_mapping, new_mm_states):
            slot_states = mm_state_rows[slot_idx]
            assert torch.all(slot_states == new_states)
