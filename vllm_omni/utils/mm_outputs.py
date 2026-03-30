"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def build_mm_cpu(multimodal_outputs: dict, seq_len: int | None) -> dict[str, object]:
    """Pre-copies multimodal tensor to CPU once (not per-request) to avoid
    redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.

    In the case of prefix caching, the multimodal outputs provided will
    only contain the passthrough data.

    Args:
        multimodal_outputs: Multimodal dict mapping strings to objects.
        seq_len: Optional sequence length (i.e., dim 0 of hidden states).
            This should be set to None in the prefix caching case since we
            only consider the passtrhough data.
    """
    # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
    # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
    mm_cpu: dict[str, object] = {}
    if multimodal_outputs:
        for k, v in multimodal_outputs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == seq_len:
                mm_cpu[k] = v.detach().to("cpu").contiguous()
            elif isinstance(v, dict):
                sub_dict: dict[str, torch.Tensor] = {}
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor) and sv.shape[0] == seq_len:
                        sub_dict[str(sk)] = sv.detach().to("cpu").contiguous()
                if sub_dict:
                    mm_cpu[k] = sub_dict
            elif isinstance(v, list) and len(v) > 0:
                cpu_list = []
                for elem in v:
                    if isinstance(elem, torch.Tensor):
                        cpu_list.append(elem.detach().to("cpu").contiguous())
                    else:
                        cpu_list.append(elem)
                mm_cpu[k] = cpu_list
    return mm_cpu


def to_payload_element(element, idx: int, start: int, end: int, seq_len: int | None = None):
    """Build an mm payload element corresponding to one request index
    from an element containing 0 or more CPU tensors.

    Args:
        element: The object to be added to the payload.
        idx: The index of the request.
        start: The start index corresponding to the request idx.
        end: The end index corresponding to the request idx.
        seq_len: Optional sequence length (i.e., dim 0 of hidden states).
            This should be set to None in the prefix caching case, because
            the condition that would be executed here is the same as the
            criteria for being added to the multimodal outputs cache.
    """
    # Prefix cache won't hit this case because this is the considition
    # for being a mm_cache_key in the multimodal outputs tensor.
    if seq_len is not None and isinstance(element, torch.Tensor) and element.shape[0] == seq_len:
        return element[start:end].contiguous()
    # Every other case is shared between prefix cache (passthrough data)
    # and running a model without prefix caching.
    elif isinstance(element, dict):
        return {sk: sv[start:end].contiguous() for sk, sv in element.items()}
    elif isinstance(element, list):
        element = element[idx] if idx < len(element) else element[0]
        # Clone tensors to avoid cross-request aliasing
        if isinstance(element, torch.Tensor):
            element = element.clone()
        return element
    elif isinstance(element, torch.Tensor):
        # List-derived tensor payloads are request-invariant; clone to
        # avoid accidental cross-request aliasing on downstream mutation.
        return element.clone()
    return element
