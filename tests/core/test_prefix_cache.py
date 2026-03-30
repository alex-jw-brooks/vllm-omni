from unittest.mock import patch

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32
DEFAULT_SHAPE = torch.Size([NUM_BLOCKS, BLOCK_SIZE, HIDDEN_SIZE])


class MockInputBatch:
    def __init__(self, num_computed_tokens_cpu):
        self.req_ids = ["req1", "req2"]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_computed_tokens_cpu = num_computed_tokens_cpu


def build_cache_with_mm_keys(mm_cache_keys) -> OmniTensorPrefixCache:
    with patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._resolve_mm_cache_keys",
        return_value=mm_cache_keys,
    ):
        # Model config is only used for resolving the mm_cache_keys,
        # so the value passed here doesn't matter since it's patched.
        return OmniTensorPrefixCache(
            num_blocks=NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            hidden_size=HIDDEN_SIZE,
            dtype=DTYPE,
            model_config=None,
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
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    new_hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

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
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    mm_outputs = {key: torch.rand((num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in mm_cache_keys}
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
        new_mm_outputs = mm_outputs[mm_key]
        for slot_idx, new_output in zip(slot_mapping, new_mm_outputs):
            slot_states = mm_state_rows[slot_idx]
            assert torch.all(slot_states == new_output)


### Tests for Merging
def fake_get_cached_block_ids(self, req_idx, *args, **kwargs):
    """Fake block table lookup.

    Assumption:
        req_idx 0 is a cache hit with slots 8, 9, ..., 15
        req_idx 1 is a cache miss
    """
    assert req_idx < 2
    if req_idx == 0:
        # With the slot offset we provided (8), the corresponding
        # blocks IDs are 2 & 3 because the block size is 4.
        return torch.tensor([2, 3], dtype=torch.long)
    return torch.tensor([], dtype=torch.long)


def test_get_merged_hidden_states():
    """Ensure that hidden states are merged correctly."""
    cache = build_cache_with_mm_keys(mm_cache_keys=None)

    orig_num_tokens_unpadded = 8
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ..., 15
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    orig_hidden_states = torch.rand((orig_num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=orig_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
    )

    # Say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }
    new_hidden_states = torch.rand(
        (num_new_toks_req1 + num_new_toks_req2, HIDDEN_SIZE),
        dtype=DTYPE,
    )
    req1_new_states = new_hidden_states[:num_new_toks_req1]
    req2_new_states = new_hidden_states[-num_new_toks_req2:]

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    with patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    ):
        merged_states = cache.get_merged_hidden_states(
            query_start_loc=[0, num_new_toks_req1],
            input_batch=input_batch,
            hidden_states=new_hidden_states,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    assert "req1" in merged_states and "req2" in merged_states
    req1_merged_states = merged_states["req1"]
    req2_merged_states = merged_states["req2"]

    # First, check the cache hit case
    assert req1_merged_states.shape == torch.Size([orig_num_tokens_unpadded + num_new_toks_req1, HIDDEN_SIZE])
    # Ensure that the req1 merged states are the cached states + the new req1 states
    assert torch.all(req1_merged_states[:orig_num_tokens_unpadded] == orig_hidden_states)
    assert torch.all(req1_merged_states[-num_new_toks_req1:] == req1_new_states)

    # Next, ensure that the cache miss case only has the new states
    assert req2_merged_states.shape == torch.Size([num_new_toks_req2, HIDDEN_SIZE])
    assert torch.all(req2_merged_states == req2_new_states)


@pytest.mark.parametrize(
    "mm_cache_keys",
    [
        ("foo", "bar"),  # All same feature dim (HIDDEN_SIZE)
        {"foo": 100, "bar": 50, "baz": None},  # different feature dims
    ],
)
def test_get_merged_multimodal_outputs(mm_cache_keys):
    cache = build_cache_with_mm_keys(mm_cache_keys)

    orig_num_tokens_unpadded = 8
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ..., 15
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    orig_mm_outputs = {
        key: torch.rand((orig_num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in mm_cache_keys
    }

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=orig_mm_outputs,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
    )

    # Similar to hs test- say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }

    new_mm_outputs = {}
    for mm_key in mm_cache_keys:
        new_mm_outputs[mm_key] = torch.rand(
            (num_new_toks_req1 + num_new_toks_req2, feature_dims[mm_key]),
            dtype=DTYPE,
        )
    # We also want to make sure passthrough data (outside of our keys) isn't dropped
    new_mm_outputs["passthrough_data"] = "Something else"

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    with patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    ):
        merged_mm_outputs = cache.get_merged_multimodal_states(
            query_start_loc=[0, num_new_toks_req1],
            input_batch=input_batch,
            multimodal_outputs=new_mm_outputs,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    # Ensure the passthrough data wasn't dropped
    assert "passthrough_data" in merged_mm_outputs

    for mm_key, mm_output in merged_mm_outputs.items():
        # Ensure passthrough data is just forwarded normally and not duplicated
        if mm_key == "passthrough_data":
            assert new_mm_outputs[mm_key] == mm_output
            assert new_mm_outputs[mm_key] == mm_output
        else:
            assert mm_key in mm_cache_keys
            assert isinstance(mm_output, dict)
            assert "req1" in mm_output and "req2" in mm_output
            curr_feat_dim = feature_dims[mm_key]
            # Ensure that req1 (cache hit) merged the mm data
            req1_merged_mm_outputs = mm_output["req1"]
            req1_new_mm_outputs = new_mm_outputs[mm_key][:num_new_toks_req1]

            assert req1_merged_mm_outputs.shape == torch.Size(
                [orig_num_tokens_unpadded + num_new_toks_req1, curr_feat_dim]
            )
            # Ensure that the req1 merged mm data are the cached data + the new data
            assert torch.all(req1_merged_mm_outputs[:orig_num_tokens_unpadded] == orig_mm_outputs[mm_key])
            assert torch.all(req1_merged_mm_outputs[-num_new_toks_req1:] == req1_new_mm_outputs)

            # Ensure that req2 (cache miss) only has the new mm data
            req2_merged_mm_outputs = mm_output["req2"]
            req2_new_mm_outputs = new_mm_outputs[mm_key][-num_new_toks_req2:]

            assert req2_merged_mm_outputs.shape == torch.Size([num_new_toks_req2, curr_feat_dim])
            assert torch.all(req2_merged_mm_outputs == req2_new_mm_outputs)
