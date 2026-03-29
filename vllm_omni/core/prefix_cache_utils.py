"""
Utilities for Prefix Caching in Omni models.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

"""
NOTE for tomorrow - working on pulling the equivalent functionality out here. also, we should make sure
that we clear new requests, since I don't think we currently are + make sure that we pass the combined
multimodal states since we are not right now
"""

# TODO - Make this configurable and factor in the number
# of multimodal tensors.
NUM_GPU_BLOCKS = 2048
# TODO Make this generic, these are specific for qwen3 omni.
OMNI_HS_CACHE_KEY = "_OMNI_HIDDEN_STATES_KEY"
MM_CACHE_KEYS = ["0", "24"]
CACHEABLE_KEYS = MM_CACHE_KEYS + [OMNI_HS_CACHE_KEY]


class OmniTensorPrefixCache:
    """Prefix cache for hidden states (model outputs)
    and model specific multimodal outputs.

    This class implements prefix caching in a non-invasive
    way on top of vLLM by leveraging the same slot mappings
    that the vLLM scheduler uses for the KV Cache

    Conceptually, we are vLLM's mapping from:
            (num_blocks, block_size)

    and translate it to rows in the 3D tensor of shape:
            (num_blocks, block_size, feature_size)

    The above is generally a large tensor, especially since
    multiple multimodal output tensors may be cached. To reduce
    GPU pressure, we implement this as a two-tier cache, where the
    keeping the large tensor offloaded on CPU, and transferring hot
    blocks to a GPU buffer of size
            (num_gpu_blocks, block_size, feature_size)
    while maintaining a layer of indirection mapping the indices of
    the CPU slot mappings to the GPU slot mappings.
    """

    def __init__(self, num_blocks: int, block_size: int, hidden_size: int, dtype, device):
        self.new_req_cache_hit_ids: set[str] | None = None

        self.block_size = block_size
        # TODO: Support CPU offload and combine these.
        self.omni_tensor_cache = {
            cacheable_key: torch.zeros(
                (num_blocks, block_size, hidden_size),
                dtype=dtype,
                device=device,
            )
            for cacheable_key in CACHEABLE_KEYS
        }

        self._new_req_cache_hit_ids: set[str] | None = None

    @property
    def hidden_states_cache(self) -> torch.Tensor:
        """Returns the hidden states cache."""
        return self.omni_tensor_cache[OMNI_HS_CACHE_KEY]

    @property
    def mm_outputs_cache(self) -> dict[str, torch.Tensor]:
        """Returns the model specific multimodal outputs cache."""
        return {k: v for k, v in self.omni_tensor_cache.items() if k != OMNI_HS_CACHE_KEY}

    def update_omni_tensor_prefix_cache(self, hidden_states, multimodal_outputs, num_tokens_unpadded, input_batch):
        """Updates the hidden cache state for for hidden states and multimodal outputs."""
        assert self.hidden_states_cache is not None
        slot_mapping = input_batch.block_table[0].slot_mapping.gpu[:num_tokens_unpadded]
        # View the cache as 2D so that we can treat our slots as row indices
        flat_cache = self.hidden_states_cache.view(-1, self.hidden_states_cache.shape[-1])
        flat_cache[slot_mapping] = hidden_states[:num_tokens_unpadded]
        logger.info(f"[HS Cache WRITE] tokens={num_tokens_unpadded}")

        # Do the same for the cached multimodal outputs for this stage;
        # for now we assume that all of the multimodal outputs cached
        # are exactly the same size as the hidden states.
        # TODO (Alex) make this more flexible.
        if self.mm_outputs_cache is not None:
            for mm_out_key, mm_cache in self.mm_outputs_cache.items():
                assert mm_out_key in multimodal_outputs
                mm_state = multimodal_outputs[mm_out_key]
                flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
                flat_cache[slot_mapping] = mm_state[:num_tokens_unpadded]
            logger.info(f"[multimodal output Cache WRITE] tokens={num_tokens_unpadded}")

    def _get_combined_states(
        self, query_start_loc, input_batch, hidden_states, multimodal_outputs, num_scheduled_tokens
    ):
        combined_mm_states = self._get_merged_multimodal_states(
            query_start_loc, input_batch, multimodal_outputs, num_scheduled_tokens
        )
        combined_hidden_states = self._get_merged_hidden_states(
            query_start_loc, input_batch, hidden_states, num_scheduled_tokens
        )
        return combined_hidden_states, combined_mm_states

    def _get_merged_multimodal_states(self, query_start_loc, input_batch, multimodal_outputs, num_scheduled_tokens):
        """Get the merged multimodal states if hidden state prefix caching is enabled."""
        combined_multimodal_outputs = {}
        for mm_key in MM_CACHE_KEYS:
            if mm_key in multimodal_outputs:
                combined_multimodal_outputs[mm_key] = self._get_merged_tensors(
                    query_start_loc=query_start_loc,
                    input_batch=input_batch,
                    cache=self.mm_outputs_cache[mm_key],
                    hidden_states=multimodal_outputs[mm_key],
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            else:
                logger.error("Cacheable multimodal key %s is not present in multimodal outputs", mm_key)
        return combined_multimodal_outputs

    def _get_merged_hidden_states(self, query_start_loc, input_batch, hidden_states, num_scheduled_tokens):
        return self._get_merged_tensors(
            query_start_loc=query_start_loc,
            input_batch=input_batch,
            cache=self.hidden_states_cache,
            hidden_states=hidden_states,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    def _get_merged_tensors(
        self, query_start_loc, input_batch, cache: torch.Tensor, hidden_states: torch.Tensor, num_scheduled_tokens
    ) -> dict[str, torch.Tensor]:
        """When hidden state caching is enabled, takes the input hidden_states,
        which only correspond to the scheduled tokens, and returns a mapping
        from request IDs to their full hidden states. This is accomplished by
        looking up the block IDs & scheduled token counts to split the
        hidden_states.

        NOTE: We do not handle hybrid caches at the moment, which is why
        we index into the first block table like this.
        """
        combined_hidden_states = {}
        if cache is not None and self._new_req_cache_hit_ids:
            for req_id in self._new_req_cache_hit_ids:
                req_idx = input_batch.req_id_to_index[req_id]
                num_computed = input_batch.num_computed_tokens_cpu[req_idx]
                # NOTE: vLLM only caches full blocks
                num_cached_blocks = num_computed // self.block_size
                # Get the block IDs attached to this cache hit and reindex into
                # the flattened cached hidden states (i.e., 1 row per token).
                block_ids = input_batch.block_table[0].block_table.gpu[req_idx, :num_cached_blocks]
                cached_hs = cache[block_ids].reshape(-1, cache.shape[-1])

                # Slice the hidden states corresponding to this request;
                # we do this by using the query start
                start = query_start_loc.gpu[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]
                # TODO: consider putting the actually hidden state cache on CPU
                combined_hidden_states[req_id] = torch.cat([cached_hs, new_hs], dim=0)

                logger.info(
                    f"[Cache combine] req={req_id} cached_blocks={num_cached_blocks} "
                    f"cached hidden states shape={cached_hs.shape} "
                    f"new hidden states shape={new_hs.shape}"
                )

        return combined_hidden_states
