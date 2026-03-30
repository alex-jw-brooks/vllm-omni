"""
Utilities for Prefix Caching in Omni models.
"""

from typing import TypeAlias

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_omni.config.model import OmniModelConfig

logger = init_logger(__name__)


# TODO - Make this configurable and factor in the number
# of multimodal tensors.
NUM_GPU_BLOCKS = 2048

StageMMCacheKeys: TypeAlias = list[str] | dict[str, int | None]
ModelMMCacheKeys: TypeAlias = dict[str, StageMMCacheKeys] | None


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

    Currently all tensors are stored on device.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        model_config: OmniModelConfig,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.default_hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        # TODO: Support CPU offload
        self.mm_cache_keys = self._resolve_mm_cache_keys(model_config)
        self._initialize_omni_tensor_caches(self.mm_cache_keys)
        self._new_req_cache_hit_ids: set[str] = set()

    def _resolve_mm_cache_keys(self, model_config: OmniModelConfig) -> StageMMCacheKeys | None:
        """Determined the configuration for multimodal caching for the current model
        architecture and stage."""
        model_stage = model_config.model_stage
        arch, arch_str = get_model_architecture(model_config)
        if hasattr(arch, "_model_mm_cache_keys"):
            model_mm_cache_keys = arch._model_mm_cache_keys
            if model_stage in model_mm_cache_keys:
                stage_mm_cache_keys = model_mm_cache_keys[model_stage]
                logger.info(f"Resolved mm_cache_keys for stage {model_stage} - {stage_mm_cache_keys}")
                return stage_mm_cache_keys

        # TODO: Move have_multimodal_outputs to class property and set this log to
        # error level & to only go off if we actually have mm outputs.
        logger.warning(
            f"Model architecture {arch_str} does not have defined _mm_cache_keys and will"
            " therefore not able leverage prefix caching for multimodal outputs. "
            " As such, prefix caching may not be supported."
        )

    def _initialize_omni_tensor_caches(self, mm_cache_keys: StageMMCacheKeys | None):
        """Initialize the Omni Tensor cache tensors; this handles both the
        hidden states cache and the multimodal outputs cache.

        The hidden_states cache is a tensor with shape:
                (num_blocks, block_size, self.default_hidden_size)

        While the mm_outputs_cache is dict mapping keys to tensors of shape:
                (num_blocks, block_size, feature_size)

        By default, if mm_cache_keys is a list, feature_size is set to the
        default hidden size for all mm_output_keys. We also accept a dict
        mapping to feature sizes on a per key basis, falling back to
        self.default_hidden_size. for any keys that are None.
        """
        self.hidden_states_cache = self._get_cache_tensor()

        self.mm_outputs_cache = {}
        if mm_cache_keys:
            if isinstance(mm_cache_keys, dict):
                for cache_key, hidden_size in mm_cache_keys.items():
                    self.mm_outputs_cache[cache_key] = self._get_cache_tensor(
                        hidden_size=hidden_size,
                    )
            else:
                for cache_key in mm_cache_keys:
                    self.mm_outputs_cache[cache_key] = self._get_cache_tensor()

    def _get_cache_tensor(self, hidden_size: int | None = None) -> torch.Tensor:
        """Allocate a cache tensor for a specific key."""
        actual_hidden_size = hidden_size if hidden_size is not None else self.default_hidden_size
        return torch.zeros(
            (self.num_blocks, self.block_size, actual_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )

    def add_prefix_cached_new_req_id(self, req_id: str):
        """Adds a new request ID to the set of prefix cache hits on the batch."""
        self._new_req_cache_hit_ids.add(req_id)

    def reset_prefix_cached_new_req_ids(self):
        """Clears the cache hit IDs to prepare for a new engine step."""
        self._new_req_cache_hit_ids.clear()

    def update_omni_tensor_prefix_cache(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict[str, torch.Tensor] | None,
        num_tokens_unpadded: int,
        slot_mapping: torch.Tensor,
    ):
        """Updates the hidden cache state for the provided hidden states and multimodal outputs."""
        unpadded_slot_mapping = slot_mapping[:num_tokens_unpadded]
        if hidden_states is not None:
            # View the cache as 2D so that we can treat our slots as row indices
            flat_cache = self.hidden_states_cache.view(-1, self.hidden_states_cache.shape[-1])
            flat_cache[unpadded_slot_mapping] = hidden_states[:num_tokens_unpadded]
            logger.debug("Writing to hidden states for %s tokens", num_tokens_unpadded)

        # Do the same for the stage's cached multimodal outputs
        if multimodal_outputs is not None:
            for mm_out_key, mm_cache in self.mm_outputs_cache.items():
                assert mm_out_key in multimodal_outputs
                mm_state = multimodal_outputs[mm_out_key]
                flat_cache = mm_cache.view(-1, mm_cache.shape[-1])
                flat_cache[unpadded_slot_mapping] = mm_state[:num_tokens_unpadded]
            logger.debug("Writing to mm output cache for %s tokens", num_tokens_unpadded)

    def _get_combined_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ):
        combined_mm_states = self._get_merged_multimodal_states(
            query_start_loc, input_batch, multimodal_outputs, num_scheduled_tokens
        )
        combined_hidden_states = self._get_merged_hidden_states(
            query_start_loc, input_batch, hidden_states, num_scheduled_tokens
        )
        return combined_hidden_states, combined_mm_states

    def _get_merged_multimodal_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ):
        """Get the merged multimodal states if hidden state prefix caching is enabled."""
        combined_multimodal_outputs = {}
        # TODO Ensure non cached keys are properly handled.
        if self.mm_cache_keys is not None:
            for mm_key in self.mm_cache_keys:
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
        elif multimodal_outputs:
            logger.warning(
                " A model stage produced multimodal outputs, but has no defined mm_cache_keys; "
                " this probably means that prefix caching is not fully supported for all stages "
                "in this model"
            )
        return combined_multimodal_outputs

    def _get_merged_hidden_states(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ):
        return self._get_merged_tensors(
            query_start_loc=query_start_loc,
            input_batch=input_batch,
            cache=self.hidden_states_cache,
            hidden_states=hidden_states,
            num_scheduled_tokens=num_scheduled_tokens,
        )

    def _get_merged_tensors(
        self,
        query_start_loc: torch.Tensor,
        input_batch: InputBatch,
        cache: torch.Tensor,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
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
                start = query_start_loc[req_idx]
                new_hs = hidden_states[start : start + num_scheduled_tokens[req_id]]
                combined_hidden_states[req_id] = torch.cat([cached_hs, new_hs], dim=0)

                logger.info(
                    f"[Cache combine] req={req_id} cached_blocks={num_cached_blocks} "
                    f"cached hidden states shape={cached_hs.shape} "
                    f"new hidden states shape={new_hs.shape}"
                )

        return combined_hidden_states
