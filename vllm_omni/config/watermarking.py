# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm_omni.watermarking import WATERMARKER_REGISTRY


@dataclass
class WatermarkConfig:
    # NOTE: This will need to map to an object in some cases, e.g., for text coming from vLLM
    modality_algorithms: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for modality, algorithm in self.modality_algorithms.items():
            if not isinstance(modality, str):
                raise ValueError(f"watermark modality must be a string, got {type(modality).__name__}")
            registered_algorithms = WATERMARKER_REGISTRY.get(modality)
            if registered_algorithms is None:
                supported_modalities = ", ".join(sorted(WATERMARKER_REGISTRY))
                raise ValueError(f"unsupported watermark modality {modality}; supported: {supported_modalities}")
            if not isinstance(algorithm, str) or algorithm not in registered_algorithms:
                valid_algorithms = ", ".join(sorted(registered_algorithms))
                raise ValueError(
                    f"unsupported watermark algorithm {algorithm} for {modality}; supported: {valid_algorithms}"
                )
