"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def build_mm_cpu(multimodal_outputs, seq_len: int | None) -> dict[str, object]:
    # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
    # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
    mm_cpu: dict[str, object] = {}
    if isinstance(multimodal_outputs, dict) and multimodal_outputs:
        for k, v in multimodal_outputs.items():
            try:
                if isinstance(v, torch.Tensor) and v.shape[0] == seq_len:
                    mm_cpu[k] = v.detach().to("cpu").contiguous()
                elif isinstance(v, dict):
                    sub_dict: dict[str, torch.Tensor] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor) and sv.shape[0] == seq_len:
                            sub_dict[str(sk)] = sv.detach().to("cpu").contiguous()
                    if sub_dict:
                        mm_cpu[k] = sub_dict
                elif isinstance(v, list):
                    if len(v) == 0:
                        continue
                    cpu_list = []
                    for elem in v:
                        if isinstance(elem, torch.Tensor):
                            cpu_list.append(elem.detach().to("cpu").contiguous())
                        else:
                            cpu_list.append(elem)
                    mm_cpu[k] = cpu_list
            except Exception as e:
                logger.error(f"Error in merge multimodal outputs: {e}")
    return mm_cpu


def to_payload_element(element, idx, start, end, seq_len: int | None = None):
    """Given"""
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
