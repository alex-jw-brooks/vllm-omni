from unittest.mock import Mock, patch

import pytest
import torch
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved
from vllm.v1.core.kv_cache_utils import BlockHash, ExternalBlockHash

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache, TailHash
from vllm_omni.core.sched.output import OmniNewRequestData

DEFAULT_SEQ_LEN = 15
NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32
OTHER_DTYPE = torch.float16
DEFAULT_SHAPE = torch.Size([NUM_BLOCKS, BLOCK_SIZE, HIDDEN_SIZE])
MOCK_MEDIUM = "MOCK_MEDIUM"
MOCK_END_IDX = 1


class MockInputBatch:
    def __init__(self, num_computed_tokens_cpu):
        self.req_ids = ["req1", "req2"]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_computed_tokens_cpu = num_computed_tokens_cpu
        # Block table is only mocked for validation of length;
        # we don't actually need to add valid values here since
        # we patch the table when testing.
        self.block_table = Mock()
        self.block_table.block_tables = [None]


def create_scheduled_new_req_with_tail(
    req_id: str,
    num_tokens: int,
    block_size: int,
) -> list:
    """Creates scheduled_new_reqs with tail metadata for testing."""
    # The number of tokens is divisible by block size, so there is not tail stranded
    assert num_tokens % block_size != 0

    # NOTE: we add one here since at runtime we have a hash for the
    # partial block, it's just not usable for prefix caching
    num_blocks = (num_tokens // block_size) + 1
    mock_block_hashes = [f"hash_block_{i}".encode() for i in range(num_blocks)]

    # Create a fake multimodal feature
    mock_mm_feature = Mock()
    mock_mm_feature.mm_position = Mock()
    mock_mm_feature.mm_position.offset = 0
    mock_mm_feature.mm_position.length = num_tokens
    mock_mm_feature.mm_hash = f"image_hash_{num_tokens}"

    # Create a fake OmniNewRequestData wrapipng the hashes / features
    mock_req_data = Mock(spec=OmniNewRequestData)
    mock_req_data.req_id = req_id
    mock_req_data.block_hashes = mock_block_hashes
    mock_req_data.mm_features = [mock_mm_feature]

    return [mock_req_data]


def get_omni_pcache_with_mm_tensors(feat_dims, seq_len) -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache and init mm tensors."""
    cache = get_omni_pcache()
    mm_outputs = get_multimodal_outputs(feat_dims, seq_len)
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, seq_len)
    return cache


def get_omni_pcache() -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache, but don't init mm tensors."""
    cache = OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=DTYPE,
    )
    return cache


def get_multimodal_outputs(feat_dims: dict[str, int], seq_len: int) -> dict[str, torch.Tensor]:
    fake_mm_inputs = {}
    for mm_key, feat_dim in feat_dims.items():
        fake_mm_inputs[mm_key] = torch.rand((seq_len, feat_dim), dtype=DTYPE)
    return fake_mm_inputs


### Tests for initialization
def test_initialization_simple():
    """Check default initialization only creates the hidden states."""
    cache = get_omni_pcache()
    assert isinstance(cache.hidden_states_cache, torch.Tensor)
    assert cache.hidden_states_cache.shape == DEFAULT_SHAPE
    assert len(cache.mm_outputs_cache) == 0
    assert len(cache.mm_cache_keys) == 0


def test_initialization_with_multimodal():
    """Check initialization + registration of multimodal outputs."""
    cache = get_omni_pcache()
    feat_dims = {"foo": 100, "bar": 50, "baz": 10}
    mm_outputs = get_multimodal_outputs(
        feat_dims,
        seq_len=DEFAULT_SEQ_LEN,
    )
    # Cast one of the keys to a different dtype; the dtype of the tensor
    # that is used to initialize the cache dictates the cache dtype.
    mm_outputs["foo"] = mm_outputs["foo"].to(OTHER_DTYPE)

    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 3
    assert set(cache.mm_cache_keys) == set(feat_dims.keys())
    for mm_key in cache.mm_cache_keys:
        cache_tensor = cache.mm_outputs_cache[mm_key]
        assert isinstance(cache_tensor, torch.Tensor)
        assert cache_tensor.shape[-1] == feat_dims[mm_key]
        assert mm_outputs[mm_key].dtype == cache_tensor.dtype


def test_init_missing_mm_cache_keys_is_idempotent():
    """Ensure that the cache doesn't reinitialize old keys."""
    cache = get_omni_pcache()
    mm_key = "foo"
    feat_dims = {mm_key: 100}
    mm_outputs = get_multimodal_outputs(
        feat_dims,
        seq_len=DEFAULT_SEQ_LEN,
    )
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 1
    assert mm_key in cache.mm_cache_keys

    # Cache is initialized to 0 - fill it with 1s
    cache.mm_outputs_cache[mm_key].fill_(1)

    # Ensure that running another initialization
    # doesn't zero out our cache values
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 1
    assert mm_key in cache.mm_cache_keys
    assert torch.all(cache.mm_outputs_cache[mm_key] == 1)


### Tests for Update
def test_update_no_multimodal():
    """Test that slot mappings act as row indices hidden states."""
    cache = get_omni_pcache()

    num_tokens_unpadded = 8
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    new_hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=new_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
        scheduled_new_reqs=[],
        num_tokens_padded=num_tokens_unpadded,  # no padding
        query_start_loc=torch.tensor([0]),
        input_batch=MockInputBatch(num_computed_tokens_cpu=torch.tensor([0])),
    )

    # Ensure that if we reshape our 3D cache back to 2D, we can use the
    # indices in our slot mappings to access the hidden states as expected
    hs_rows = cache.hidden_states_cache.view(NUM_BLOCKS * BLOCK_SIZE, HIDDEN_SIZE)
    for slot_idx, new_states in zip(slot_mapping, new_hidden_states):
        slot_states = hs_rows[slot_idx]
        assert torch.all(slot_states == new_states)


@pytest.mark.parametrize(
    "feat_dims",
    [
        {"foo": 100, "bar": 100},
        {"foo": 100, "bar": 50, "baz": 10},
    ],
)
def test_update_with_multimodal_outputs(feat_dims):
    """Test that slot mappings are correct for multimodal tensors."""
    cache = get_omni_pcache_with_mm_tensors(feat_dims, seq_len=DEFAULT_SEQ_LEN)

    num_tokens_unpadded = 8
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    mm_outputs = {key: torch.rand((num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys}
    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=mm_outputs,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
        scheduled_new_reqs=[],
        num_tokens_padded=num_tokens_unpadded,  # no padding
        query_start_loc=torch.tensor([0]),
        input_batch=MockInputBatch(num_computed_tokens_cpu=torch.tensor([0])),
    )

    for mm_key in feat_dims.keys():
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


@pytest.mark.parametrize("tail_len", [0, 1])
@pytest.mark.parametrize("num_tokens_padded", [None, 16])
def test_get_merged_hidden_states(tail_len, num_tokens_padded):
    """Ensure that hidden states are merged correctly."""
    cache = get_omni_pcache()

    # If it's a partial block, add an extral prompt token (since block size is 4).
    orig_num_tokens_unpadded = 8 + tail_len
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ..., 15
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    orig_hidden_states = torch.rand((orig_num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    # Say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3 + tail_len
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

    # Create tail metadata for partial block case
    scheduled_new_reqs = (
        create_scheduled_new_req_with_tail("req1", orig_num_tokens_unpadded, BLOCK_SIZE) if tail_len else []
    )

    cache.update_omni_tensor_prefix_cache(
        hidden_states=orig_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
        scheduled_new_reqs=scheduled_new_reqs,
        num_tokens_padded=num_tokens_padded if num_tokens_padded else orig_num_tokens_unpadded,
        query_start_loc=torch.tensor([0]),
        input_batch=input_batch,
    )

    with patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    ):
        merged_states = cache.get_merged_hidden_states(
            query_start_loc=[0, num_new_toks_req1],
            input_batch=input_batch,
            hidden_states=new_hidden_states,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_new_reqs=scheduled_new_reqs,
        )

    assert "req1" in merged_states and "req2" in merged_states
    req1_merged_states = merged_states["req1"]
    req2_merged_states = merged_states["req2"]

    # First, check the cache hit case
    assert req1_merged_states.shape == torch.Size(
        [orig_num_tokens_unpadded + num_new_toks_req1 - tail_len, HIDDEN_SIZE]
    )
    # Ensure that the req1 merged states are the cached states + the new req1 states
    assert torch.all(req1_merged_states[:orig_num_tokens_unpadded] == orig_hidden_states)
    assert torch.all(req1_merged_states[-num_new_toks_req1:] == req1_new_states)

    # Next, ensure that the cache miss case only has the new states
    assert req2_merged_states.shape == torch.Size([num_new_toks_req2, HIDDEN_SIZE])
    assert torch.all(req2_merged_states == req2_new_states)


@pytest.mark.parametrize("tail_len", [0, 1])
@pytest.mark.parametrize("num_tokens_padded", [None, 16])
@pytest.mark.parametrize(
    "feat_dims",
    [
        {"foo": 100, "bar": 100},
        {"foo": 100, "bar": 50, "baz": 10},
    ],
)
def test_get_merged_multimodal_outputs(tail_len, feat_dims, num_tokens_padded):
    cache = get_omni_pcache_with_mm_tensors(feat_dims, seq_len=DEFAULT_SEQ_LEN)

    # If it's a partial block, add an extral prompt token (since block size is 4).
    orig_num_tokens_unpadded = 8 + tail_len
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ...
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    orig_mm_outputs = {
        key: torch.rand((orig_num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys
    }

    # Similar to hs test- say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3 + tail_len
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }

    new_mm_outputs = {}
    for mm_key in cache.mm_cache_keys:
        new_mm_outputs[mm_key] = torch.rand(
            (num_new_toks_req1 + num_new_toks_req2, feature_dims[mm_key]),
            dtype=DTYPE,
        )
    # We also want to make sure passthrough data (outside of our keys) isn't dropped
    new_mm_outputs["passthrough_data"] = "Something else"
    # Lists are a special case because we can't split them yet if we want to match
    # the nonprefix cache behavior, because this runs before post process.
    new_mm_outputs["passthrough_list"] = ["should", "not", "split"]

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    # Create tail metadata for partial block case
    scheduled_new_reqs = (
        create_scheduled_new_req_with_tail("req1", orig_num_tokens_unpadded, BLOCK_SIZE) if tail_len else []
    )

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=orig_mm_outputs,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
        scheduled_new_reqs=scheduled_new_reqs,
        num_tokens_padded=num_tokens_padded if num_tokens_padded else orig_num_tokens_unpadded,
        query_start_loc=torch.tensor([0]),
        input_batch=input_batch,
    )

    with patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    ):
        merged_mm_outputs = cache.get_merged_multimodal_states(
            query_start_loc=[0, num_new_toks_req1],
            input_batch=input_batch,
            multimodal_outputs=new_mm_outputs,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_new_reqs=scheduled_new_reqs,
        )

    # Ensure the passthrough data wasn't dropped
    assert "passthrough_data" in merged_mm_outputs
    assert "passthrough_list" in merged_mm_outputs

    for mm_key, mm_output in merged_mm_outputs.items():
        # Ensure passthrough data is just forwarded normally and not duplicated
        assert isinstance(mm_output, dict)
        assert "req1" in mm_output and "req2" in mm_output
        if mm_key == "passthrough_data":
            assert mm_key not in cache.mm_cache_keys
            assert new_mm_outputs[mm_key] == mm_output["req1"]
            assert new_mm_outputs[mm_key] == mm_output["req2"]
        elif mm_key == "passthrough_list":
            assert mm_key not in cache.mm_cache_keys
            assert new_mm_outputs[mm_key] == mm_output["req1"]
            assert new_mm_outputs[mm_key] == mm_output["req2"]
        else:
            assert mm_key in cache.mm_cache_keys
            curr_feat_dim = feature_dims[mm_key]
            # Ensure that req1 (cache hit) merged the mm data
            req1_merged_mm_outputs = mm_output["req1"]
            req1_new_mm_outputs = new_mm_outputs[mm_key][:num_new_toks_req1]

            assert req1_merged_mm_outputs.shape == torch.Size(
                [orig_num_tokens_unpadded + num_new_toks_req1 - tail_len, curr_feat_dim]
            )
            # Ensure that the req1 merged mm data are the cached data + the new data
            assert torch.all(req1_merged_mm_outputs[:orig_num_tokens_unpadded] == orig_mm_outputs[mm_key])
            assert torch.all(req1_merged_mm_outputs[-num_new_toks_req1:] == req1_new_mm_outputs)

            # Ensure that req2 (cache miss) only has the new mm data
            req2_merged_mm_outputs = mm_output["req2"]
            req2_new_mm_outputs = new_mm_outputs[mm_key][-num_new_toks_req2:]

            assert req2_merged_mm_outputs.shape == torch.Size([num_new_toks_req2, curr_feat_dim])
            assert torch.all(req2_merged_mm_outputs == req2_new_mm_outputs)


### Tests for Tail Caching Eviction Behaviors
# NOTE: Multimodal tail caching is the main dynamically allocated component
# in omni prefix caching, since the part corresponding to the main vLLM prefix
# cache currently allocates CPU tensors up front to map to the block pool,
# assuming most multimodal keys are known up front. As such, we need to
# have solid eviction tests to ensure that we don't leak memory.
def test_tail_eviction_empty_event_list():
    """Ensure that we don't evict existing tails if we have no events."""
    cache = get_omni_pcache()

    # Cache a tail
    block_hash = BlockHash(b"block_hash_0")
    tail_len = 1
    tail_hash = TailHash("image_hash_0", tail_len, MOCK_END_IDX)
    mock_tail = torch.rand((tail_len, HIDDEN_SIZE), dtype=DTYPE)
    cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, mock_tail, None)

    # Handle empty event list
    cache.handle_kv_cache_events([])

    # Verify tail still exists
    assert block_hash in cache.block_tail_cache.hidden_states_cache
    block_hashed_tails = cache.block_tail_cache.hidden_states_cache[block_hash]
    # The block should still link the tail hash to our tail tensor
    assert len(block_hashed_tails) == 1
    retrieved_tail = block_hashed_tails[tail_hash]
    assert isinstance(retrieved_tail, torch.Tensor)
    assert torch.all(retrieved_tail == mock_tail)


@pytest.mark.parametrize(
    "del_indices",
    [
        [1],  # One block evicted
        [1, 3, 4],  # Multiple blocks evicted
    ],
)
def test_tail_eviction_on_block_removed(del_indices):
    """Test that tails are evicted when KV cache blocks are removed."""
    cache = get_omni_pcache()

    # Create and populate tail cache with 5 blocks
    num_blocks = 5
    block_hashes = [BlockHash(f"block_hash_{i}".encode()) for i in range(num_blocks)]
    tail_hashes = []
    mock_tails = []
    for i, block_hash in enumerate(block_hashes):
        tail_hash = TailHash(f"image_hash_{i}", i + 1, MOCK_END_IDX)
        mock_tail = torch.rand((i + 1, HIDDEN_SIZE), dtype=DTYPE)
        tail_hashes.append(tail_hash)
        mock_tails.append(mock_tail)
        cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, mock_tail, None)

    # Verify all tails exist
    assert len(cache.block_tail_cache.hidden_states_cache) == num_blocks

    # Emit BlockRemoved event for specified blocks and update tail cache
    hashes_to_evict: list[ExternalBlockHash] = [block_hashes[i] for i in del_indices]
    event = BlockRemoved(block_hashes=hashes_to_evict, medium=MOCK_MEDIUM)
    cache.handle_kv_cache_events([event])

    # Ensure we dropped the caches for hashes at our del_indices, but kept the rest
    expected_remaining = num_blocks - len(del_indices)
    assert len(cache.block_tail_cache.hidden_states_cache) == expected_remaining

    for i, (mock_tail, block_hash, tail_hash) in enumerate(zip(mock_tails, block_hashes, tail_hashes)):
        if i in del_indices:
            assert block_hash not in cache.block_tail_cache.hidden_states_cache
        else:
            block_hashed_tails = cache.block_tail_cache.hidden_states_cache[block_hash]
            retrieved_tail = block_hashed_tails[tail_hash]
            assert torch.all(retrieved_tail == mock_tail)


@pytest.mark.parametrize("use_mm_outputs", [False, True])
def test_tail_eviction_multiple_tails_per_block(use_mm_outputs):
    """Ensure that if a block has multiple tails, block eviction clears them all."""
    if use_mm_outputs:
        cache = get_omni_pcache_with_mm_tensors({"foo": 100, "bar": 50}, seq_len=DEFAULT_SEQ_LEN)
        cache_dict = cache.block_tail_cache.mm_outputs_cache
    else:
        cache = get_omni_pcache()
        cache_dict = cache.block_tail_cache.hidden_states_cache

    # Create a single block with multiple tails
    block_hash = BlockHash(b"block_hash_shared")
    tail_hashes = [TailHash(f"image_hash_{i}", i + 1, MOCK_END_IDX) for i in range(3)]

    for tail_hash in tail_hashes:
        tail_len = tail_hash.tail_len
        if use_mm_outputs:
            mm_tails = {
                "foo": torch.rand((tail_len, 100), dtype=DTYPE),
                "bar": torch.rand((tail_len, 50), dtype=DTYPE),
            }
            cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, None, mm_tails)
        else:
            mock_tail = torch.rand((tail_len, HIDDEN_SIZE), dtype=DTYPE)
            cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, mock_tail, None)

    # Verify all tails are stored under the same block
    assert block_hash in cache_dict
    assert len(cache_dict[block_hash]) == 3
    for tail_hash in tail_hashes:
        assert tail_hash in cache_dict[block_hash]

    # Evict the block - should remove all tails
    evict_hashes: list[ExternalBlockHash] = [block_hash]
    event = BlockRemoved(block_hashes=evict_hashes, medium=MOCK_MEDIUM)
    cache.handle_kv_cache_events([event])

    # Verify entire block entry was removed
    assert block_hash not in cache_dict


def test_tail_eviction_all_blocks_cleared():
    """Test that all tails are evicted when all blocks are cleared."""
    cache = get_omni_pcache_with_mm_tensors({"foo": 100}, seq_len=DEFAULT_SEQ_LEN)
    num_blocks = 3
    # Cache multiple tails for both hidden states and multimodal outputs
    for i in range(num_blocks):
        block_hash = BlockHash(f"block_hash_{i}".encode())
        tail_hash = TailHash(f"image_hash_{i}", i + 1, MOCK_END_IDX)

        # Add tails for both the hidden states and mm outputs
        mock_hs_tail = torch.rand((i + 1, HIDDEN_SIZE), dtype=DTYPE)
        cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, mock_hs_tail, None)

        mock_mm_tails = {"foo": torch.rand((i + 1, 100), dtype=DTYPE)}
        cache.block_tail_cache._cache_tail_tensors(block_hash, tail_hash, None, mock_mm_tails)

    # Verify tails exist for both HS / multimodal
    assert len(cache.block_tail_cache.hidden_states_cache) == num_blocks
    assert len(cache.block_tail_cache.mm_outputs_cache) == num_blocks

    # Verify handling AllBlocksCleared deletes all tails for every hash
    event = AllBlocksCleared()
    cache.handle_kv_cache_events([event])
    assert len(cache.block_tail_cache.hidden_states_cache) == 0
    assert len(cache.block_tail_cache.mm_outputs_cache) == 0


def test_tail_eviction_no_op_on_missing_blocks():
    """Test that evicting missing blocks doesn't cause bad behavior."""
    # NOTE: This shouldn't really happen for models with full support,
    # but we should ensure prefix cache tail tracking never crashes and
    # at most warns (as it's possible in the future depending on hybrid cache
    # handling etc).
    cache = get_omni_pcache()

    # Cache a tail for block_0
    block_hash_0 = BlockHash(b"block_hash_0")
    tail_hash = TailHash("image_hash_0", 1, MOCK_END_IDX)
    mock_tail = torch.rand((1, HIDDEN_SIZE), dtype=DTYPE)
    cache.block_tail_cache._cache_tail_tensors(block_hash_0, tail_hash, mock_tail, None)

    # Verify tail exists
    assert block_hash_0 in cache.block_tail_cache.hidden_states_cache

    # Ensure evicting block hashes not in the tail cache does
    # not crash or affect the previously cached tails
    non_existent: list[ExternalBlockHash] = [b"block_hash_99", b"block_hash_100"]
    event = BlockRemoved(block_hashes=non_existent, medium=MOCK_MEDIUM)
    cache.handle_kv_cache_events([event])

    # Verify original tail is still intact
    assert block_hash_0 in cache.block_tail_cache.hidden_states_cache
    assert tail_hash in cache.block_tail_cache.hidden_states_cache[block_hash_0]
