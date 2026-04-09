"""
Utilities for Prefix Caching in Omni models.
"""

from dataclasses import dataclass

import torch
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, KVCacheEvent
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

logger = init_logger(__name__)


@dataclass(frozen=True)
class TailHash:
    """Hashable key for mapping tails."""

    mm_hash: str
    tail_len: int
    mm_end: int


class BlockTailCache:
    """Partial block cache for caching outputs for multimodal tails. This is a separate
    cache which caches the outputs corresponding to multimodal inputs that do not
    cleanly land on a block boundary. We need to do this so that we can correctly
    look up the cached states for multimodal inputs, since otherwise the results
    would be truncated since vLLM doesn't cache partial blocks.

    Example: Say we have a block size of 8. If we receive 15 inputs, where 10
    correspond to image features, and the last 5 correspond to text, we have the
    following scenario.

    Block 1: | I I I I I I I I |
    Block 2: | I I T T T T T   |

    Block 1 is a full block, so we can cache the hidden states / multimodal outputs.
    However, since Block 2 is not a full block, vLLM wont cache it. This puts the
    first two placeholders of block 2 in a strange position with respect to prefix
    caching, because it often doesn't make sense to consider chunks of multimodal
    data in this way.

    The BlockTailCache handles this case and stores the 'tail' part tied to
    a block sequence wherein multimodal data is truncated. This is accomplished
    by mapping the hash of the parent block (i.e., Block 1) to the corresponding
    tail tensors. For example, say we had the following after processing the
    above (where I represents the same image data):

    Block 1: | I I I I I I I I | < Prefix hit
    Block 2: | I I             | < Prefix miss
    -> 8 tokens scheduled, but because we have the MM feature spilling over,
       look up the outputs with the Hash of the last parent block (Block 1),
       and concatenate the results, thereby getting the full prefix from the
       multimodal data.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.hidden_states_cache: dict[BlockHash, dict[TailHash, torch.Tensor]] = {}
        self.mm_outputs_cache: dict[BlockHash, dict[TailHash, dict[str, torch.Tensor]]] = {}

    def evict_block_tails_for_hash_list(self, block_hashes: list[BlockHash]):
        """Given a list of block hashes, delete the corresponding cache tensors."""
        for block_hash in block_hashes:
            if block_hash in self.hidden_states_cache:
                del self.hidden_states_cache[block_hash]
            if block_hash in self.mm_outputs_cache:
                del self.mm_outputs_cache[block_hash]

    def evict_all_block_tails(self):
        """Clear all block tails."""
        self.hidden_states_cache.clear()
        self.mm_outputs_cache.clear()

    def get_request_mm_tail_hashes(
        self,
        req: OmniNewRequestData,
    ) -> dict[BlockHash, TailHash]:
        """Given a request, extract the tail hashes for any multimodal inputs
        that don't end on exact block boundaries.

        Returns a dict mapping parent block hashes (i.e., last complete block
        overlapping the MM object) to the tail hash.
        """
        req_block_tails = {}
        for feat in req.mm_features:
            # Should not happen, but if it does, raise, since this is an invariant
            # of the tail cache implementation to avoid recomputing hashes
            if feat.mm_hash is None:
                raise RuntimeError("Multimodal input has no hash; cannot save tail cache")

            mm_end = feat.mm_position.offset + feat.mm_position.length
            end_block = mm_end // self.block_size

            # Get the indices of the complete blocks this feature touches.
            # If the end index falls past the edge of the last complete block,
            # we have to handle the tail.
            tail_len = mm_end - (end_block * self.block_size)

            # We don't cache if we start and end in the first block or have no tail.
            if end_block == 0 or tail_len == 0 or feat.mm_hash is None:
                continue

            parent_hash = req.block_hashes[end_block]
            tail_hash = TailHash(feat.mm_hash, tail_len, mm_end)
            req_block_tails[parent_hash] = tail_hash
        return req_block_tails

    def get_cache_tail(
        self,
        req_id: str,
        block_hashes: list[BlockHash],
        mm_features: list[MultiModalFeatureSpec],
        input_batch,
        mm_key: str | None = None,
    ) -> torch.Tensor | None:
        """Given a request, retrieve the tail tensors corresponding to the multimodal
        object (if any) spills over the cache boundary.

        NOTE: the call to this can be optimized a bit in the future to avoid looping
        over mm features for every mm key + 1, but it's probably not needed for now
        since the number of input multimodal features is usually small.
        """
        # Find cache boundary using unpadded token count
        req_idx = input_batch.req_id_to_index[req_id]
        num_computed = input_batch.num_computed_tokens_cpu[req_idx]

        # Calculate tail position at cache boundary
        end_block = int(num_computed // self.block_size)
        tail_len = int(num_computed % self.block_size)
        cacheable_end_idx = end_block * self.block_size

        # Early exit cases - we have no MM features or end cleanly on a full block
        if tail_len == 0 or not mm_features:
            return None

        # Find which MM feature ends at the cache boundary
        for feat in mm_features:
            mm_start = feat.mm_position.offset
            mm_end = feat.mm_position.offset + feat.mm_position.length
            # The object starts in a cached prefix and ends after our final block
            if mm_start < num_computed and mm_end > cacheable_end_idx:
                if feat.mm_hash is None:
                    raise RuntimeError("Multimodal input has no hash; cannot retrieve cached tail")

                parent_hash = block_hashes[end_block]
                tail_hash = TailHash(feat.mm_hash, tail_len, mm_end)

                # Tail cache hit on hidden states
                if mm_key is None and parent_hash in self.hidden_states_cache:
                    return self.hidden_states_cache[parent_hash].get(tail_hash)

                # Look up the single tail - from mm_outputs_cache or hidden_states_cache
                if (
                    mm_key is not None
                    and parent_hash in self.mm_outputs_cache
                    and tail_hash in self.mm_outputs_cache[parent_hash]
                ):
                    return self.mm_outputs_cache[parent_hash][tail_hash].get(mm_key)
            # Multimodal objects are sorted, so once we pass the boundary
            # for our last cached block, we won't ever have a tail cache hit
            elif mm_start > num_computed:
                return None

    def cache_block_tail(
        self,
        new_req: OmniNewRequestData,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict | None,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        mm_cache_keys: set[str],
    ) -> None:
        """Cache tail portions for requests with partial multimodal blocks."""
        # Identify tails and their positions
        req_block_tail_info = self.get_request_mm_tail_hashes(new_req)

        # Either we have no multimodal inputs or it all ends at exact block boundaries
        if not req_block_tail_info:
            return

        # Get the token offset for this request in the batch
        req_idx = input_batch.req_id_to_index[new_req.req_id]
        token_offset = int(query_start_loc[req_idx])

        for parent_hash, tail_hash in req_block_tail_info.items():
            # Calculate absolute indices of the tail start/end. This is tied to a
            # single multimodal object, so we may have multiple in one request
            start_idx = token_offset + (tail_hash.mm_end - tail_hash.tail_len)
            end_idx = token_offset + tail_hash.mm_end

            # Get the tail tensors and store them in the corresponding cache
            hs_tail = (
                OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states[start_idx:end_idx])
                if hidden_states is not None
                else None
            )

            mm_tails = (
                self._extract_mm_tails(multimodal_outputs, start_idx, end_idx, mm_cache_keys)
                if multimodal_outputs
                else None
            )

            # Store the cached tensors corresponding to the current tail
            self._cache_tail_tensors(parent_hash, tail_hash, hs_tail, mm_tails)

    def _extract_mm_tails(
        self,
        multimodal_outputs: dict,
        start_idx: int,
        end_idx: int,
        mm_cache_keys: set[str],
    ) -> dict[str, torch.Tensor]:
        """Extract multimodal tails corresponding to a single multimodal object."""
        mm_tails = {}
        for mm_key, mm_tensor in multimodal_outputs.items():
            # If it's an mm cache key, we already know it's a tensor
            if mm_key in mm_cache_keys:
                mm_tails[mm_key] = OmniTensorPrefixCache._coerce_to_cpu_tensor(mm_tensor[start_idx:end_idx])
        return mm_tails

    def _cache_tail_tensors(
        self,
        parent_hash: BlockHash,
        tail_hash: TailHash,
        hs_tail: torch.Tensor | None,
        mm_tails: dict[str, torch.Tensor] | None,
    ):
        """Given tail tensors that have already been sliced and moved to CPU,
        store the hidden states and/or multimodal tails.
        """
        # Cache hidden states
        if hs_tail is not None:
            if parent_hash not in self.hidden_states_cache:
                self.hidden_states_cache[parent_hash] = {}
            self.hidden_states_cache[parent_hash][tail_hash] = hs_tail

        # Cache multimodal outputs
        if mm_tails:
            if parent_hash not in self.mm_outputs_cache:
                self.mm_outputs_cache[parent_hash] = {}
            if tail_hash not in self.mm_outputs_cache[parent_hash]:
                self.mm_outputs_cache[parent_hash][tail_hash] = {}
            self.mm_outputs_cache[parent_hash][tail_hash].update(mm_tails)

    def handle_kv_cache_events(self, kv_cache_events: list[KVCacheEvent]):
        """Update the tail block cache to reflect the later step's evictions."""
        for event in kv_cache_events:
            if isinstance(event, BlockRemoved):
                self.evict_block_tails_for_hash_list(event.block_hashes)
            elif isinstance(event, AllBlocksCleared):
                self.evict_all_block_tails()


class OmniTensorPrefixCache:
    """Prefix cache for hidden states (model outputs) and model specific multimodal outputs.

    This class implements prefix caching in a non-invasive way on top of vLLM by leveraging
    the same slot mappings that the vLLM scheduler uses for the KV Cache, while maintaining
    a separate cache for multimodal tails to handle partial blocks.

    For the complete block case, this means we are mapping vLLM's cache mapping:
                        (num_blocks, block_size)

    to 3D tensors of shape:
                   (num_blocks, block_size, feature_size)

    Where the feature_size may vary across multimodal_outputs. For more details about
    partial block caching, see the BlockTailCache.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        hs_dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.default_hidden_size = hidden_size

        # Initialize the hidden states cache immediately
        self.hidden_states_cache = self._get_cache_tensor(dtype=hs_dtype)

        # Defer initialization of the mm_outputs_cache until we
        # actually see mm output tensors dependent on num tokens.
        self.mm_outputs_cache = {}
        self.mm_cache_keys = set()
        self._new_req_cache_hit_ids: set[str] = set()

        # Since vLLM Only tracks full blocks, we need to also track
        # cases where multimodal data ends in the final block and is
        # incomplete, otherwise we can't look up the full prefix.
        self.block_tail_cache = BlockTailCache(block_size)

    def maybe_init_missing_mm_cache_keys(self, multimodal_outputs: dict, seq_len: int):
        """Given multimodal outputs from executing the model, dynamically determine which
        multimodal outputs are tensors depending on sequence length and should be cached,
        and initialize the cache tensors accordingly.

        NOTE: This is done to avoid the need for explicit specification of cache keys for
        every model/stage and aligns with the current way that we slice the multimodal
        outputs based on the first dimension.
        """
        for key, val in multimodal_outputs.items():
            if isinstance(val, torch.Tensor) and val.shape[0] == seq_len and key not in self.mm_cache_keys:
                feat_dim = val.shape[-1]
                self.mm_outputs_cache[key] = self._get_cache_tensor(
                    dtype=val.dtype,
                    hidden_size=feat_dim,
                )
                self.mm_cache_keys.add(key)
                new_tensor_shape = self.mm_outputs_cache[key].shape
                logger.info("Initializing multimodal output cache of size %s for key: %s", list(new_tensor_shape), key)

    def _get_cache_tensor(self, dtype: torch.dtype, hidden_size: int | None = None) -> torch.Tensor:
        """Allocate a CPU cache tensor for a specific key."""
        actual_hidden_size = hidden_size if hidden_size is not None else self.default_hidden_size
        return torch.zeros(
            (self.num_blocks, self.block_size, actual_hidden_size),
            dtype=dtype,
            device="cpu",
        )

    def add_prefix_cached_new_req_id(self, req_id: str):
        """Adds a new request ID to the set of prefix cache hits on the batch."""
        self._new_req_cache_hit_ids.add(req_id)

    def reset_prefix_cached_new_req_ids(self):
        """Clears the cache hit IDs to prepare for a new engine step."""
        self._new_req_cache_hit_ids.clear()

    @staticmethod
    def _coerce_to_cpu_tensor(maybe_gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Convert GPU tensors -> contiguous CPU tensors if needed."""
        return maybe_gpu_tensor.detach().cpu().contiguous()

    def update_omni_tensor_prefix_cache(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict[str, torch.Tensor] | None,
        num_tokens_unpadded: int,
        slot_mapping: torch.Tensor,
        scheduled_new_reqs: list[OmniNewRequestData],
        num_tokens_padded: int,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
    ):
        """Updates the hidden cache state for the provided hidden states and multimodal outputs.

        Args:
            hidden_states: Hidden states tensor to cache (if any)
            multimodal_outputs: Multimodal dict whose tensors may be cached
            num_tokens_unpadded: Number of tokens without padding
            slot_mapping: Slot mapping for the input sequence
            num_tokens_padded: Total number of tokens including padding
        """
        unpadded_slot_mapping = slot_mapping[:num_tokens_unpadded]

        if num_tokens_padded is None:
            num_tokens_padded = num_tokens_unpadded

        if hidden_states is not None:
            # Slice to unpadded portion before caching
            hidden_states = hidden_states[:num_tokens_unpadded]
            # Ensure that hidden states are on the CPU
            hidden_states = OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states)
            # View the cache as 2D so that we can treat our slots as row indices
            flat_cache = self.hidden_states_cache.view(-1, self.hidden_states_cache.shape[-1])
            flat_cache[unpadded_slot_mapping] = hidden_states
            logger.debug("Writing to hidden states for %s tokens", num_tokens_unpadded)

        # Do the same for the stage's cached multimodal outputs
        if multimodal_outputs is not None:
            # If we haven't initialized the keys already, do it now
            # We check against the padded token count since we haven't sliced yet
            self.maybe_init_missing_mm_cache_keys(
                multimodal_outputs,
                seq_len=num_tokens_padded,
            )

            for mm_out_key, mm_cache in self.mm_outputs_cache.items():
                if mm_out_key in multimodal_outputs:
                    # Slice to unpadded portion before caching
                    mm_state = multimodal_outputs[mm_out_key][:num_tokens_unpadded]
                    mm_state = OmniTensorPrefixCache._coerce_to_cpu_tensor(mm_state)
                    flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
                    flat_cache[unpadded_slot_mapping] = mm_state
            logger.debug("Writing to mm output cache for %s tokens", num_tokens_unpadded)

        # Update tail caches if needed; use the unpadded sequence length since we have sliced
        for new_req in scheduled_new_reqs:
            self.block_tail_cache.cache_block_tail(
                new_req,
                hidden_states,
                multimodal_outputs,
                query_start_loc,
                input_batch,
                mm_cache_keys=self.mm_cache_keys,
            )

    def _coerce_to_payload_dict(
        self,
        element: object,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, object]:
        """Build the multimodal passthrough data per request for
        the object under consideration. This is identical to the case
        for no prefix cache when we tensor does have a first dimension
        matching the seq len.
        """
        elem_dict = {}
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]
            start = query_start_loc[req_idx]
            end = start + num_scheduled_tokens[req_id]
            elem_dict[req_id] = to_payload_element(
                element, req_idx, start=start, end=end, pass_lists_through=True, seq_len=None
            )
        return elem_dict

    def get_merged_multimodal_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
        scheduled_new_reqs: list[OmniNewRequestData],
    ):
        """Get the merged multimodal states if hidden state prefix caching is enabled."""
        combined_multimodal_outputs = {}
        # First get the prefix cached tensors that are present in the mm data
        for mm_key in self.mm_cache_keys:
            if mm_key in multimodal_outputs:
                combined_multimodal_outputs[mm_key] = self._get_merged_tensors(
                    query_start_loc=query_start_loc,
                    input_batch=input_batch,
                    cache=self.mm_outputs_cache[mm_key],
                    hidden_states=multimodal_outputs[mm_key],
                    num_scheduled_tokens=num_scheduled_tokens,
                    scheduled_new_reqs=scheduled_new_reqs,
                    mm_key=mm_key,
                )

        # Then, get everything else (passthrough data); first, convert to CPU
        # tensors similarly to the non prefix cached path, and then populate
        # the subdicts mapping request IDs -> payload objects
        passthrough_keys = set(multimodal_outputs.keys()) - self.mm_cache_keys
        passthrough_mm_data = {k: v for k, v in multimodal_outputs.items() if k in passthrough_keys}
        mm_cpu = build_mm_cpu(multimodal_outputs=passthrough_mm_data)

        for mm_key, mm_val in mm_cpu.items():
            combined_multimodal_outputs[mm_key] = self._coerce_to_payload_dict(
                element=mm_val,
                query_start_loc=query_start_loc,
                input_batch=input_batch,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_multimodal_outputs

    def get_merged_hidden_states(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Get the merged hidden states."""
        return self._get_merged_tensors(
            *args,
            **kwargs,
            cache=self.hidden_states_cache,
        )

    def _get_merged_tensors(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        cache: torch.Tensor,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
        scheduled_new_reqs: list[OmniNewRequestData],
        mm_key: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """When hidden state caching is enabled, takes the input hidden_states,
        which only correspond to the scheduled tokens, and returns a mapping
        from request IDs to their full hidden states. This is accomplished by
        looking up the block IDs & scheduled token counts to split the
        hidden_states.
        """
        # We do not support hybrid caches at the moment.
        if len(input_batch.block_table.block_tables) > 1:
            logger.warning_once(
                "Omni prefix caching is enabled, but the batch block table appears to"
                " have multiple kv groups; only the first group will be used!"
            )

        combined_hidden_states = {}
        hidden_states = OmniTensorPrefixCache._coerce_to_cpu_tensor(hidden_states)
        for req_id in input_batch.req_ids:
            req_idx = input_batch.req_id_to_index[req_id]

            if req_id in self._new_req_cache_hit_ids:
                block_ids = self._get_cached_block_ids(req_idx, input_batch)
                cached_hs = cache[block_ids].reshape(-1, cache.shape[-1])

                # Slice the hidden states corresponding to this request;
                # we do this by using the query start
                start = query_start_loc[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]

                # Lookup the cache tail tensor corresponding to partially cached mm data if we have
                # one; we need to do this to avoid dropping the features from the original request
                # on a cache hit if multimodal data spills over the edge of a cached block.
                maybe_cache_tail = None
                for new_req in scheduled_new_reqs:
                    # TODO - optimize this. We don't run the outer loop over scheduled_new_reqs
                    # since we also need to build the data for the cache miss cases.
                    if new_req.req_id == req_id:
                        maybe_cache_tail = self.block_tail_cache.get_cache_tail(
                            req_id, new_req.block_hashes, new_req.mm_features, input_batch, mm_key=mm_key
                        )
                    break

                # If we have a cache tail hit, slice it into the correct location
                # FIXME (Alex): It would be more correct to count this as a cache hit
                # upfront first, mask the inputs, and concatenate the result so that
                # we don't run the tail inputs through forward. This will also include
                # the tail in the cached token count.
                if maybe_cache_tail is not None:
                    tail_len = maybe_cache_tail.shape[0]
                    new_hs[:tail_len] = maybe_cache_tail

                combined_hidden_states[req_id] = torch.cat([cached_hs, new_hs], dim=0)

            else:
                # cache miss for this request, pass through normally
                start = query_start_loc[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]
                combined_hidden_states[req_id] = new_hs

        return combined_hidden_states

    def _get_cached_block_ids(self, req_idx: int, input_batch: InputBatch) -> torch.Tensor:
        """Given an input batch and request index in the batch (not ID), get the
        block IDs corresponding to the cache hit.
        """
        num_computed = input_batch.num_computed_tokens_cpu[req_idx]
        # NOTE: vLLM only caches full blocks
        num_cached_blocks = num_computed // self.block_size
        # Get the block IDs attached to this cache hit and reindex into
        # the flattened cached hidden states (i.e., 1 row per token).
        return input_batch.block_table[0].block_table.cpu[req_idx, :num_cached_blocks]

    def handle_kv_cache_events(self, kv_cache_events: list[KVCacheEvent]):
        """Sync events from the kv cache to the block tail cache."""
        self.block_tail_cache.handle_kv_cache_events(kv_cache_events)
