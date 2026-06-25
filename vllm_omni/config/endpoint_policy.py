# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Endpoint restriction policy for omni pipelines."""

from dataclasses import dataclass
from enum import Enum


class OmniServingCapability(Enum):
    """Serving capabilities that pipelines can restrict."""

    # TODO: We need to clarify the relationship between serving
    # capabilities and output modalities. For now this enum is
    # API level since its only used for completions to prevent server
    # crashes, but it's likely we will see a strong correlation to output
    # modality and compatible endpoints as time goes on.
    COMPLETIONS = "completions"


@dataclass(frozen=True)
class EndpointRestriction:
    capability: OmniServingCapability
    reason: str
