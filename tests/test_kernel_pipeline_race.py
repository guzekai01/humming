"""GPU regression test for the multi-stage pipeline stage-reuse race.

Guards the ``__syncthreads()`` before producer stage reuse in the
``kNumStages >= 3`` branch of ``humming/include/humming/kernel/humming.cuh``
(commit 3346f2f). Without it, a CTA with two warpgroups (block_n=256,
warp_n=32) could reload a shared-memory stage while the other warpgroup was
still consuming it: deterministic garbage under stream-k timing, a
low-probability flaky race otherwise.

Methodology (mirrors the adjudication that root-caused the bug):

- Runs through the tune measurement harness (``humming.tune.measure``), the
  same proven-trustworthy path used to adjudicate the fix (0/10 -> 10/10).
  The (slow, trusted) heuristic config is the correctness reference;
  identical inputs are guaranteed by the harness (activations seeded per
  shape_m), and a slow reference ensures every candidate gets checked.
- Race bugs make single passes meaningless, so every case repeats with a
  fresh worker process per repetition.
- block_n=128 (single warpgroup) is the negative control and must stay green
  with or without the fix; block_n=256 + stream_k exercised the hazard
  deterministically pre-fix.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()

# Faithful to the adjudicated reproducer: Kimi w13 shape, fp8 activation,
# uint4 weight, bf16 group-32 weight scale, per-token input scale, wgmma.
_META = {
    "pad_shape_n": 0,
    "pad_shape_k": 0,
    "num_experts": 384,
    "b_dtype": "uint4",
    "a_dtype": "float8e4m3",
    "c_dtype": "bfloat16",
    "bs_dtype": "bfloat16",
    "input_scale_group_size": 0,
    "weight_scale_group_size": 32,
    "weight_scale_group_size_n": 0,
    "weight_scale_type": "group",
    "use_int_weight_scale": False,
    "use_fused_e8m0_scale": False,
    "has_zero_point": False,
    "is_fp_zero_point": False,
    "has_bias": False,
    "mma_type": "wgmma",
    "shape_n": 512,
    "shape_k": 7168,
    "sublayer_name": "w13",
}
SHAPE_M = 2048
TOP_K = 8
REPETITIONS = 20


def _config(block_n: int, use_stream_k: bool) -> dict:
    return {
        "block_shape": (48, block_n, 64),
        "warp_shape": (48, 32, 64),
        "num_stages": 5,
        "num_ctas_per_sm": 2,
        "use_stream_k": use_stream_k,
        "num_sms": 78,
        "use_f16_accum": False,
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires PyTorch with a CUDA GPU")
@pytest.mark.parametrize("block_n", [128, 256])
@pytest.mark.parametrize("use_stream_k", [True, False])
def test_multistage_pipeline_stage_reuse(block_n: int, use_stream_k: bool) -> None:
    from humming.config import GemmType
    from humming.tune import get_heuristics_config, measure
    from humming.tune.measured import (
        _make_layer_args,
        _normalize_meta,
        _prepare_config,
    )

    meta = _normalize_meta(dict(_META))
    gemm_type = GemmType.INDEXED
    layer_args = _make_layer_args(
        meta,
        gemm_type,
        top_k=TOP_K,
        is_moe_down=False,
        balanced=True,
        expert_max_tokens=None,
        input_scale_group_size=0,
    )
    # The heuristic config is the trusted reference AND is slow at this m,
    # which matters: measure() only correctness-checks candidates whose coarse
    # time beats the baseline, so a fast poisoned candidate is always checked.
    baseline = _prepare_config(
        dict(
            get_heuristics_config(
                meta=meta,
                use_f16_accum=False,
                gemm_type=gemm_type,
                shape_m=SHAPE_M,
            )
        ),
        meta,
        gemm_type,
        False,
        False,
    )
    candidate = _prepare_config(
        _config(block_n, use_stream_k), meta, gemm_type, False, False
    )
    if candidate == baseline:
        pytest.skip("candidate is the reference config; self-comparison is vacuous")

    def _key(config: dict) -> str:
        return repr(sorted(config.items()))

    failures = []
    for repetition in range(REPETITIONS):
        # Fresh Measurer => fresh CUDA worker process per repetition; a race
        # gets an independent scheduling sample every time. Short rep windows:
        # only the correctness compare matters here, not timing quality.
        with measure.Measurer(
            layer_args,
            gemm_type,
            progress_log_path=(
                f"/tmp/race_test_{block_n}_{use_stream_k}_{repetition}.progress"
            ),
            coarse_rep_ms=10,
            fine_rep_ms=50,
        ) as measurer:
            timings = measurer.measure(
                measure.MeasureRequest(
                    shape_m=SHAPE_M,
                    configs=[candidate],
                    baseline_config=baseline,
                )
            )
        timing = next((t for t in timings if _key(t.config) == _key(candidate)), None)
        assert timing is not None, "candidate timing missing from measure() result"
        if timing.correctness_ok is not True:
            failures.append(
                f"repetition {repetition + 1}/{REPETITIONS}: "
                f"ok={timing.correctness_ok} fail={timing.fail_reason}"
            )

    assert not failures, (
        f"block_n={block_n} use_stream_k={use_stream_k} failed "
        f"{len(failures)}/{REPETITIONS} repetitions:\n" + "\n".join(failures[:5])
    )
