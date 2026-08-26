# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Factory for building quantization configs.

build_quantization_config() delegates to vLLM's quantization registry. Omni
configs are registered into that registry by register_omni_quantization_configs().
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from vllm.logger import init_logger


# ---------------------------------------------------------------------------
# Stub the ``humming`` package so that vLLM's lazy import inside
# ``get_quantization_config()`` (which unconditionally does
# ``from .humming import HummingConfig``) does not crash when the real
# ``humming`` wheel is not installed.  Only populate the bare-minimum
# names that ``humming.py`` accesses at module level.
# ---------------------------------------------------------------------------
def _register_humming_stubs() -> None:
    """Register stub ``humming`` sub-modules so that the optional
    humming quantization backend can be imported without the real wheel."""
    if "humming" in sys.modules:
        return  # already present (real or stub)

    # --- sub-modules ---
    submodules: dict[str, tuple[str, ...]] = {
        "humming": (),
        "humming.config": ("GemmType",),
        "humming.dtypes": ("DataType",),
        "humming.layer": ("HummingLayerMeta", "HummingMethod"),
        "humming.schema": (
            "BaseInputSchema",
            "BaseWeightSchema",
            "HummingInputSchema",
            "HummingWeightSchema",
        ),
        "humming.utils": (),
        "humming.utils.weight": ("quantize_weight",),
    }

    registry: dict[str, ModuleType] = {}
    for name, attrs in submodules.items():
        mod = ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, type(attr, (), {}))
        registry[name] = mod

    # wire parent references
    setattr(registry["humming"], "config", registry["humming.config"])
    setattr(registry["humming"], "dtypes", registry["humming.dtypes"])
    setattr(registry["humming"], "layer", registry["humming.layer"])
    setattr(registry["humming"], "schema", registry["humming.schema"])
    setattr(registry["humming"], "utils", registry["humming.utils"])
    setattr(registry["humming.utils"], "weight", registry["humming.utils.weight"])

    for name, mod in registry.items():
        sys.modules[name] = mod


_register_humming_stubs()

from vllm.model_executor.layers.quantization import (  # noqa: E402
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
)

from .component_config import ComponentQuantizationConfig  # noqa: E402

logger = init_logger(__name__)


def register_omni_quantization_configs() -> None:
    """Import omni quant config modules so their @register_quantization_config
    decorators fire. This ensures that Omni's quantization definitions are registered
    over vLLM's quantization definitions, which ensures that the same quantization
    definitions are used in vLLM's ModelConfig.get_quantization_config() path (i.e.,
    for AR) as in the diffusion factory lookup.
    """
    from . import (  # noqa: F401  (import side-effect = decorator registration)
        bitsandbytes_config,
        inc_config,
        int8_config,
        mxfp4_config,
        mxfp8_config,
    )


# Omni configs registered into vLLM's registry. Static so membership/count is
# independent of when registration fires; auto-round spellings alias to inc.
'''
TODO: We need to handle torchao builders properly


def _build_torchao(**kw: Any) -> QuantizationConfig:
    """Build a TorchAO runtime or serialized-checkpoint config."""
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

    if "quant_type" in kw:
        return TorchAOConfig.from_config({**kw, "quant_method": "torchao"})
    return TorchAOConfig(**kw)


def _build_torchao_float8_weight_only(**kw: Any) -> QuantizationConfig:
    """Build the serialized TorchAO FP8 weight-only checkpoint config."""
    from torchao.quantization import Float8WeightOnlyConfig

    return _build_torchao(
        torchao_config=Float8WeightOnlyConfig(
            set_inductor_config=False,
        ),
        is_checkpoint_torchao_serialized=True,
    )
'''
_OMNI_QUANT_METHODS = ("int8", "bitsandbytes", "mxfp8", "mxfp4", "mxfp4_dualscale", "svdquant", "inc", "auto-round", "auto_round", "torchao", "torchao_float8_weight_only")
SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys([*QUANTIZATION_METHODS, *_OMNI_QUANT_METHODS]))

_QUANT_METHOD_ALIASES = {"auto-round": "inc", "auto_round": "inc"}




def _normalize_quant_method_alias(method: str | None) -> str | None:
    """Fold known aliases (auto-round/auto_round to inc); pass everything else through."""
    if method is None:
        return None
    return _QUANT_METHOD_ALIASES.get(method, method)


_MODEL_OPT_METHODS = {
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
}
_MODEL_OPT_FP8_ALGOS = {
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
}
_MODEL_OPT_NVFP4_ALGOS = {
    "NVFP4",
}
_MODEL_OPT_MIXED_ALGOS = {
    "MIXED_PRECISION",
}


def _normalize_method_name(method: Any) -> str:
    return str(method).lower().replace("-", "_")


# TODO(Alex): vLLM's ModelOpt configs probably already detect this with
# override_quantization_method, so we may be able to leverage the upstream code for this.
def _detect_modelopt_method(config: Mapping[str, Any]) -> str | None:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        quant_algo = str(quantization.get("quant_algo", "")).upper()
    else:
        quant_algo = str(config.get("quant_algo", "")).upper()

    method = config.get("method", config.get("quant_method"))
    normalized_method = _normalize_method_name(method) if method is not None else None

    producer = config.get("producer")
    is_modelopt_config = normalized_method in _MODEL_OPT_METHODS or (
        isinstance(producer, Mapping) and str(producer.get("name", "")).lower() == "modelopt"
    )

    if not is_modelopt_config:
        return None

    if quant_algo:
        if quant_algo in _MODEL_OPT_FP8_ALGOS:
            return "modelopt"
        if quant_algo in _MODEL_OPT_NVFP4_ALGOS:
            return "modelopt_fp4"
        if quant_algo in _MODEL_OPT_MIXED_ALGOS:
            return "modelopt_mixed"
        return None

    if method is not None:
        if normalized_method in _MODEL_OPT_METHODS:
            return normalized_method

    return None


def _build_modelopt_from_config(method: str, config: Mapping[str, Any]) -> QuantizationConfig:
    config_cls = get_quantization_config(method)
    normalized_config = dict(config)
    normalized_config.setdefault("quant_method", method)
    return config_cls.from_config(normalized_config)


def _pop_method_name(spec: dict[str, Any]) -> str | None:
    method = spec.pop("method", None)
    if method is None:
        method = spec.pop("quant_method", None)
    if method is not None and not isinstance(method, str):
        raise TypeError(f"'method'/'quant_method' must be a string, got {type(method).__name__}")
    return method


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization.

    A per-component dict has no "method" / "quant_method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method" /
    "quant_method".
    """
    if "method" in spec or "quant_method" in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and ("method" in v or "quant_method" in v)) for v in spec.values())


def _maybe_build_component_quant_config(
    spec: dict[str, Any],
    quant_config: dict[str, Any] | None,
) -> ComponentQuantizationConfig | None:
    if not _is_per_component_dict(spec):
        return None
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None
    for prefix, value in spec.items():
        if not isinstance(value, (str, dict, QuantizationConfig, type(None))):
            raise TypeError(
                f"Per-component value for {prefix!r} must be str, dict, "
                f"QuantizationConfig, or None, got {type(value).__name__}"
            )
        resolved = build_quantization_config(value, quant_config)
        if prefix == "default":
            default_config = resolved
        else:
            component_configs[prefix] = resolved
    return ComponentQuantizationConfig(component_configs, default_config)


def build_quantization_config(
    quantization: str | dict[str, Any] | QuantizationConfig | None,
    quant_config: dict[str, Any] | None = None,
) -> QuantizationConfig | None:
    """Build a resolved QuantizationConfig.

    Examples::

        build_quantization_config("fp8")
        build_quantization_config("fp8", {"quant_method": "fp8", "is_checkpoint_fp8_serialized": True})
        build_quantization_config({"method": "fp8", "activation_scheme": "static"})
        build_quantization_config({"transformer": "fp8", "vae": None}) # component config

    Args:
        quantization: Method string, dict spec, QuantizationConfig passthrough, or None.
        quant_config: Checkpoint quantization metadata dict (e.g. from a model's
            config.json ``quantization_config`` field). Passed to ``from_config()``
            for checkpoint-quantized models. Omit for online quantization.
    """
    if isinstance(quantization, QuantizationConfig) or quantization is None:
        return quantization

    # If a quantization config is provided, ensure Omni's quantization configs are registered.
    register_omni_quantization_configs()

    if isinstance(quantization, Mapping):
        spec = dict(quantization)
        component_cfg = _maybe_build_component_quant_config(spec, quant_config)
        if component_cfg is not None:
            return component_cfg

        # HF checkpoint dicts carry "quant_method"; inline specs carry "method".
        # FIXME, quant_method/method shouldn't matter for choice, but currently
        # it does, which messes up int8.
        from_checkpoint = "quant_method" in spec
        quantization = _pop_method_name(spec)
        if quantization is None:
            raise ValueError(
                "Dict quantization config must have a 'method' or 'quant_method' key "
                "or be a per-component config with component prefixes as keys."
            )
        modelopt_method = _detect_modelopt_method({"quant_method": quantization, **spec})
        if modelopt_method is not None:
            return _build_modelopt_from_config(modelopt_method, {"quant_method": quantization, **spec})
    else:
        spec = dict(quant_config) if isinstance(quant_config, dict) else {}
        from_checkpoint = "quant_method" in spec

    method = _normalize_quant_method_alias(quantization)
    if method == "none":
        return None

    if method not in QUANTIZATION_METHODS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    # Checkpoint dicts go through from_config (plucks only wanted keys); inline
    # specs construct directly. Restore quant_method popped above, since some
    # from_config impls read it (e.g. int8 derives is_checkpoint_*_serialized).
    quant_cls = get_quantization_config(method)
    if from_checkpoint:
        spec.setdefault("quant_method", quantization)
        return quant_cls.from_config(spec)
    return quant_cls(**spec)


def _disk_marks_serialized(qc_kwargs: dict[str, Any], quant_config: object) -> bool:
    """Return True when config.json says serialized but the active quant_config does not.

    Matches any flag following the is_checkpoint_*_serialized naming convention,
    so new quant methods don't require updating an explicit allowlist.
    """
    for key, val in qc_kwargs.items():
        if key.startswith("is_checkpoint_") and key.endswith("_serialized"):
            if val and hasattr(quant_config, key) and not getattr(quant_config, key):
                return True
    return False


def maybe_rebuild_quantization_config(quant_config: QuantizationConfig, disk_qc: dict[str, Any]) -> QuantizationConfig:
    """Produce the final quantization config, which will either be a newly built config ,
    or a handle to the original if we can reuse it. Currently this is only applicable for
    models that need to consider that case where we may have multiple quantization configs,
    E.g., wan2_2.
    """
    qc_method: str = disk_qc["quant_method"]
    qc_kwargs: dict[str, Any] = {k: v for k, v in disk_qc.items() if k != "quant_method"}
    if _disk_marks_serialized(qc_kwargs, quant_config):
        logger.info(
            "config.json marks checkpoint as serialized; switching to offline %s mode.",
            qc_method,
        )
        return build_quantization_config(qc_method, disk_qc)

    # AutoRound MXFP8 checkpoints use data_type="mx_fp" instead of
    # is_checkpoint_*_serialized; rebuild so the offline path is selected.
    if qc_kwargs.get("data_type") == "mx_fp":
        logger.info("config.json declares data_type='mx_fp'; rebuilding as offline AutoRound MXFP8.")
        return build_quantization_config(qc_method, disk_qc)

    if (
        "ignored_layers" in qc_kwargs
        and hasattr(quant_config, "ignored_layers")
        and set(qc_kwargs.get("ignored_layers") or []) != set(quant_config.ignored_layers or [])
    ):
        logger.info("config.json ignored_layers differs from active config; rebuilding quant_config.")
        return build_quantization_config(qc_method, disk_qc)
    return quant_config


def resolve_quant_config_from_disk(
    quant_config: QuantizationConfig | None,
    disk_qc: dict[str, Any] | str | None,
) -> QuantizationConfig | None:
    """Reconcile an active quant_config against a transformer's config.json.

    Used for cascade models where individual transformer blocks have their
    own config.json (e.g. separate transformer and transformer_2 directories).
    Returns the disk config when it carries more specific info than the active one.
    """
    if disk_qc is None:
        return quant_config

    if quant_config is None:
        return build_quantization_config(disk_qc, disk_qc if isinstance(disk_qc, dict) else None)

    if isinstance(disk_qc, str):
        return quant_config

    if not isinstance(disk_qc, Mapping) or "quant_method" not in disk_qc:
        return quant_config

    disk_method = _normalize_quant_method_alias(disk_qc["quant_method"])
    active_method = _normalize_quant_method_alias(quant_config.get_name())
    if active_method != disk_method:
        raise ValueError(
            f"Checkpoint config.json declares quant_method={disk_qc['quant_method']!r} "
            f"but the active quantization config is {quant_config.get_name()!r}."
        )

    return maybe_rebuild_quantization_config(quant_config, disk_qc)
