# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MoE backend selection for models with SwiGLU alpha/beta/limit params.

GPU-free: mocks the device gate, then exercises ``is_supported_config`` and the
fp8 oracle so FlashInfer TRTLLM is skipped (with fallback) for fp8 block-scale
checkpoints with SwiGLU params (e.g. DeepSeek V4), which the pinned FlashInfer
version rejects at runtime and older versions silently drop (#53411), while
MXFP8 checkpoints with SwiGLU params (e.g. gpt-oss) still select it.
"""

import dataclasses
from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
)

_EXPERTS_DIR = "vllm.model_executor.layers.fused_moe.experts"


@pytest.fixture(autouse=True)
def _neutral_env(monkeypatch):
    """Selection reads these on the host; unset them so assertions are
    deterministic across CI runners."""
    for key in (
        "VLLM_USE_DEEP_GEMM",
        "VLLM_MOE_USE_DEEP_GEMM",
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_BATCH_INVARIANT",
    ):
        monkeypatch.delenv(key, raising=False)


def _config(
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation: MoEActivation = MoEActivation.SILU,
    routing_method: RoutingMethodType = RoutingMethodType.DeepSeekV3,
):
    cfg = make_dummy_moe_config(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=7168,
        activation=activation,
    )
    return dataclasses.replace(
        cfg,
        routing_method=routing_method,
        router_logits_dtype=torch.float32,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


def _blackwell_trtllm():
    """Pass the SM100-family device gate off-hardware."""
    return patch(
        f"{_EXPERTS_DIR}.trtllm_fp8_moe.TrtLlmFp8ExpertsBase._supports_current_device",
        return_value=True,
    )


def _blackwell_oracle_platform():
    """Make the oracle's platform-based priority ordering deterministic
    (Blackwell CUDA) regardless of the host running the test."""
    return patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform",
        **{
            "is_cuda.return_value": True,
            "is_rocm.return_value": False,
            "is_cpu.return_value": False,
            "is_xpu.return_value": False,
            "is_device_capability.return_value": False,
            "is_device_capability_family.return_value": True,
        },
    )


@pytest.mark.parametrize(
    "experts_cls", [TrtLlmFp8ExpertsModular, TrtLlmFp8ExpertsMonolithic]
)
@pytest.mark.parametrize(
    "swiglu_params",
    [
        {"swiglu_limit": 10.0},
        {"swiglu_alpha": 1.702},
        {"swiglu_beta": 1.0},
    ],
)
def test_block_scale_with_swiglu_params_rejected(experts_cls, swiglu_params):
    """fp8 block-scale + any SwiGLU param (DeepSeek V4 sets swiglu_limit):
    TRTLLM must be rejected, since the pinned FlashInfer only applies the
    params for MXFP8."""
    with _blackwell_trtllm():
        ok, reason = experts_cls.is_supported_config(
            experts_cls,
            _config(**swiglu_params),
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is False
    assert "SwiGLU" in reason


@pytest.mark.parametrize(
    "experts_cls", [TrtLlmFp8ExpertsModular, TrtLlmFp8ExpertsMonolithic]
)
def test_block_scale_without_swiglu_params_supported(experts_cls):
    """fp8 block-scale without SwiGLU params (DeepSeek V3): still selectable."""
    with _blackwell_trtllm():
        ok, reason = experts_cls.is_supported_config(
            experts_cls,
            _config(),
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is True
    assert reason is None


@pytest.mark.parametrize(
    "experts_cls", [TrtLlmFp8ExpertsModular, TrtLlmFp8ExpertsMonolithic]
)
def test_mxfp8_with_swiglu_params_supported(experts_cls):
    """MXFP8 + SwiGLU alpha/beta/limit (gpt-oss): FlashInfer applies the
    params for MXFP8, so TRTLLM must stay selectable."""
    with _blackwell_trtllm():
        ok, reason = experts_cls.is_supported_config(
            experts_cls,
            _config(
                swiglu_limit=7.0,
                swiglu_alpha=1.702,
                swiglu_beta=1.0,
                activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                routing_method=RoutingMethodType.Renormalize,
            ),
            kMxfp8Static,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is True
    assert reason is None


def test_per_tensor_with_swiglu_params_not_rejected():
    """The guard is scoped to the block-scale scheme: per-tensor fp8 with
    SwiGLU params keeps its current selection behavior (that path does not
    plumb the params at all — a pre-existing gap outside this guard)."""
    with _blackwell_trtllm():
        ok, reason = TrtLlmFp8ExpertsMonolithic.is_supported_config(
            TrtLlmFp8ExpertsMonolithic,
            _config(swiglu_limit=10.0),
            kFp8StaticTensorSym,
            kFp8StaticTensorSym,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is True
    assert reason is None


def _mock_non_trtllm_backends():
    """Force a deterministic oracle fallback chain regardless of host platform:
    AITER/FI-CUTLASS unusable, DEEPGEMM usable."""
    return (
        patch(
            f"{_EXPERTS_DIR}.rocm_aiter_moe.AiterExperts.is_supported_config",
            return_value=(False, "mocked out"),
        ),
        patch(
            f"{_EXPERTS_DIR}.flashinfer_cutlass_moe.FlashInferExperts"
            ".is_supported_config",
            return_value=(False, "mocked out"),
        ),
        patch(
            f"{_EXPERTS_DIR}.triton_deep_gemm_moe.TritonOrDeepGemmExperts"
            ".is_supported_config",
            return_value=(True, None),
        ),
    )


def test_auto_selects_trtllm_without_swiglu_limit():
    """Auto mode control: without a clamp, TRTLLM wins on Blackwell."""
    aiter, fi_cutlass, deepgemm = _mock_non_trtllm_backends()
    with _blackwell_trtllm(), _blackwell_oracle_platform(), aiter, fi_cutlass, deepgemm:
        backend, _ = select_fp8_moe_backend(
            _config(),
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )
    assert backend == Fp8MoeBackend.FLASHINFER_TRTLLM


def test_auto_falls_back_with_swiglu_limit():
    """Auto mode with a clamp: TRTLLM is skipped and selection falls through
    to a clamp-capable backend instead of silently dropping the clamp."""
    aiter, fi_cutlass, deepgemm = _mock_non_trtllm_backends()
    with _blackwell_trtllm(), _blackwell_oracle_platform(), aiter, fi_cutlass, deepgemm:
        backend, _ = select_fp8_moe_backend(
            _config(swiglu_limit=10.0),
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )
    assert backend == Fp8MoeBackend.DEEPGEMM


def test_explicit_trtllm_with_swiglu_limit_raises():
    """--moe-backend flashinfer_trtllm with a clamp: clear error instead of
    FlashInfer's kernel-level ValueError (or silently corrupted output)."""
    cfg = dataclasses.replace(
        _config(swiglu_limit=10.0), moe_backend="flashinfer_trtllm"
    )
    with (
        _blackwell_trtllm(),
        _blackwell_oracle_platform(),
        pytest.raises(ValueError, match="SwiGLU"),
    ):
        select_fp8_moe_backend(
            cfg,
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )
