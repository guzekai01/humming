#!/usr/bin/env python3
"""Compile and validate K3 M32 native-fused versus mode2-fixed1+C.

This is a correctness/compile harness, not a latency benchmark.  Arms A and C
launch the same transformed resident tensors; their only difference is the
experimental TuningConfig bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

import humming
from humming import ops
from humming.config import GemmType
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayer
from humming.testing.data import generate_moe_tensors, generate_random_topk_ids

NUM_EXPERTS = 896
ROUTER_TOP_K = 16
TOKEN_M = 32
ROUTED_M = TOKEN_M * ROUTER_TOP_K
BLOCK_M = 8
STAGES = {
    "w13": {
        "shape_n": 768,
        "shape_k": 3584,
        "kernel_top_k": 16,
        "historical_config_id": "92ac5c500b03",
        "tuning": {
            "block_shape": (8, 256, 64),
            "warp_shape": (8, 64, 64),
            "use_stream_k": True,
            "num_sms": 78,
            "num_stages": 3,
            "num_ctas_per_sm": 3,
        },
    },
    "w2": {
        "shape_n": 3584,
        "shape_k": 384,
        "kernel_top_k": 1,
        "historical_config_id": "fe708102472d",
        "tuning": {
            "block_shape": (8, 128, 128),
            "warp_shape": (8, 32, 128),
            "use_stream_k": False,
            "num_sms": 78,
            "num_stages": 5,
            "num_ctas_per_sm": 4,
        },
    },
}
RTOL = 0.01
ATOL = 0.05


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--route-seed", type=int, default=20260819)
    parser.add_argument(
        "--w13-sk0-control",
        action="store_true",
        help="keep the exact W13 geometry but disable Stream-K for correctness attribution",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    # Byte-view first so this also works for bfloat16 and float8 tensors whose
    # NumPy dtype may not exist in the host environment.  Stream large K3
    # weights in bounded chunks rather than materializing another multi-GB
    # host-side bytes object.
    values = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024 * 1024
    for offset in range(0, values.numel(), chunk_bytes):
        chunk = values[offset : offset + chunk_bytes].cpu().numpy().tobytes()
        digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _new_layer(shape_n: int, shape_k: int) -> HummingLayer:
    return HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=NUM_EXPERTS,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 32,
            "scale_dtype": "float8e8m0",
            "has_zero_point": False,
            "is_fp_zero_point": False,
        },
        input_config={"dtype": "float8e4m3", "group_size": 0},
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")


def _fill_raw_parameters(layer: HummingLayer) -> None:
    """Fill packed FP4 codes and well-conditioned canonical E8M0 scales."""

    with torch.no_grad():
        for name, tensor in layer.named_parameters():
            if name == "weight":
                tensor.random_(-(2**31), 2**31 - 1)
            elif name == "weight_scale" and str(tensor.dtype).startswith("torch.float8_e8m0"):
                # Keep the raw-reference gate well conditioned.  The decoder
                # unit test separately exhausts every stored r in [1, 12].
                tensor.view(torch.uint8).random_(125, 130)
            elif "scale" in name and tensor.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                tensor.uniform_(0.5, 2.0)
            else:
                raise RuntimeError(f"unexpected raw parameter {name}: {tensor.dtype}")


def _snapshot_raw(layer: HummingLayer) -> tuple[dict[str, torch.Tensor], dict]:
    raw = {name: value.detach().clone() for name, value in layer.named_parameters()}
    hashes = {name: _tensor_sha256(value) for name, value in raw.items()}
    return raw, hashes


def _raw_reference(layer: HummingLayer, raw: dict[str, torch.Tensor]) -> torch.Tensor:
    weight_ref = layer.weight_schema.dequant_tensors(raw).float()
    if not torch.isfinite(weight_ref).all():
        raise RuntimeError("raw dequantized weight reference is not finite")
    return weight_ref


def _resident_manifest(layer: HummingLayer) -> dict:
    result = {}
    for name in ("weight", "weight_scale", "weight_scale_2"):
        tensor = getattr(layer, name)
        result[name] = {
            "ptr": tensor.data_ptr(),
            "sha256": _tensor_sha256(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    stored_scale = layer.weight_scale.view(torch.uint8)
    result["stored_relative_scale_range"] = {
        "min": int(stored_scale.min().item()),
        "max": int(stored_scale.max().item()),
    }
    scale_range = result["stored_relative_scale_range"]
    if not (1 <= scale_range["min"] <= scale_range["max"] <= 12):
        raise RuntimeError(f"bad stored scale range: {result['stored_relative_scale_range']}")
    return result


def _historical_config_id(tuning: dict) -> str:
    # The 2026-08-19 sweep carried use_f16_accum in the raw heuristic dict;
    # today it remains a ComputeConfig field.  Reinsert it only for provenance.
    historical = dict(tuning)
    historical.pop("use_mode2_fixed_mxfp4_c_scale", None)
    historical["use_f16_accum"] = False
    return _json_sha256(historical)[:12]


def _arm_tuning(stage: str, *, fixed: bool, w13_sk0_control: bool) -> dict:
    tuning = dict(STAGES[stage]["tuning"])
    if w13_sk0_control:
        if stage != "w13":
            raise ValueError("--w13-sk0-control requires --stage w13")
        tuning["use_stream_k"] = False
    tuning["use_mode2_fixed_mxfp4_c_scale"] = fixed
    return tuning


def _compile_arm(
    layer: HummingLayer,
    stage: str,
    arm: str,
    *,
    fixed: bool,
    w13_sk0_control: bool,
) -> dict:
    assert layer.humming_config is not None
    compute = {
        "use_f16_accum": False,
        "use_m_major_input_scale": False,
        "gemm_type": GemmType.INDEXED.value,
    }
    tuning = _arm_tuning(stage, fixed=fixed, w13_sk0_control=w13_sk0_control)
    compute_str = json.dumps(compute, sort_keys=True)
    tuning_str = json.dumps(tuning, sort_keys=True)
    config_table = HummingKernel.prepare_kernels(
        layer.humming_config.to_str(),
        compute_str,
        tuning_str,
    ).reshape(-1, 4)
    if config_table.shape[0] != 1:
        raise RuntimeError(f"{arm} compiled {config_table.shape[0]} kernels")
    configs = config_table[0]
    kernel_id = int(configs[2].item())
    kernel = HummingKernel._id2kernel[kernel_id]
    kernel.assert_smem_size_matches_estimate()
    cubin = Path(kernel.kernel_filename).resolve()
    if not cubin.is_file():
        raise RuntimeError(f"{arm} cubin is missing: {cubin}")
    return {
        "arm": arm,
        "fixed": fixed,
        "compute": compute,
        "tuning": tuning,
        "tuning_key_sha256": _sha256_bytes(tuning_str.encode()),
        "configs": configs,
        "kernel_id": kernel_id,
        "kernel_name": kernel.kernel_name,
        "cubin": str(cubin),
        "cache_dir": str(cubin.parent),
        "cubin_sha256": _sha256_bytes(cubin.read_bytes()),
        "estimated_smem_size": kernel.estimated_smem_size,
    }


def _check_distinct_compiles(arm_a: dict, arm_c: dict) -> None:
    checks = {
        "tuning_key": arm_a["tuning_key_sha256"] != arm_c["tuning_key_sha256"],
        "kernel_id": arm_a["kernel_id"] != arm_c["kernel_id"],
        "cache_dir": arm_a["cache_dir"] != arm_c["cache_dir"],
        "cubin_path": arm_a["cubin"] != arm_c["cubin"],
        "cubin_sha256": arm_a["cubin_sha256"] != arm_c["cubin_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"A/C did not compile distinctly: {failed}")


def _routing(route_seed: int) -> tuple[torch.Tensor, dict, dict]:
    torch.cuda.manual_seed(route_seed)
    topk_ids = generate_random_topk_ids(TOKEN_M, NUM_EXPERTS, ROUTER_TOP_K)
    _, _, sorted_ids, expert_ids, num_tokens_padded = generate_moe_tensors(
        topk_ids,
        NUM_EXPERTS,
        GemmType.INDEXED,
        block_size_config=BLOCK_M,
    )
    counts = torch.bincount(topk_ids.flatten(), minlength=NUM_EXPERTS)
    active_experts = int((counts > 0).sum().item())
    expert_m_tiles = int(((counts + BLOCK_M - 1) // BLOCK_M).sum().item())
    issued_m = int(num_tokens_padded.item())
    if issued_m != expert_m_tiles * BLOCK_M:
        raise RuntimeError("routing metadata has inconsistent issued-M")
    if route_seed == 20260819 and (active_experts, expert_m_tiles, issued_m) != (
        383,
        383,
        3064,
    ):
        raise RuntimeError(
            "default route no longer matches the exact sweep anchor: "
            f"{active_experts}/{expert_m_tiles}/{issued_m}"
        )
    metadata = {
        "topk_ids": topk_ids,
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_tokens_padded": num_tokens_padded,
    }
    manifest = {
        "route_seed": route_seed,
        "active_experts": active_experts,
        "expert_m_tiles": expert_m_tiles,
        "issued_m": issued_m,
        "topk_ids_sha256": _tensor_sha256(topk_ids),
        "sorted_ids_sha256": _tensor_sha256(sorted_ids),
        "expert_ids_sha256": _tensor_sha256(expert_ids),
    }
    return topk_ids, metadata, manifest


def _quantized_inputs(stage: str, shape_k: int, seed: int) -> tuple[torch.Tensor, ...]:
    input_rows = TOKEN_M if stage == "w13" else ROUTED_M
    torch.cuda.manual_seed(seed + 100_000)
    hidden = torch.randn(
        (input_rows, shape_k),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    inputs, input_scale = ops.quant_input(
        hidden,
        "float8e4m3",
        group_size=0,
        scale_dtype="float32",
        m_major_scale=False,
    )
    scale_ref = input_scale.float()
    if scale_ref.ndim == 1:
        scale_ref = scale_ref.unsqueeze(1)
    inputs_ref = inputs.float() * scale_ref
    return inputs, input_scale, inputs_ref


def _fp32_reference(
    stage: str,
    inputs_ref: torch.Tensor,
    weight_ref: torch.Tensor,
    topk_ids: torch.Tensor,
    shape_n: int,
) -> torch.Tensor:
    flat_experts = topk_ids.flatten().long()
    token_inputs = inputs_ref.repeat_interleave(ROUTER_TOP_K, dim=0) if stage == "w13" else inputs_ref
    output = torch.empty((ROUTED_M, shape_n), dtype=torch.float32, device="cuda:0")
    for expert_id in flat_experts.unique(sorted=True).tolist():
        token_ids = torch.where(flat_experts == expert_id)[0]
        values = token_inputs[token_ids].float().matmul(weight_ref[expert_id].float().T)
        output[token_ids] = values
    if not torch.isfinite(output).all():
        raise RuntimeError("FP32 oracle is not finite")
    return output.to(torch.bfloat16)


def _run_arm(
    layer: HummingLayer,
    arm: dict,
    stage: str,
    inputs: torch.Tensor,
    input_scale: torch.Tensor,
    metadata: dict,
    shape_n: int,
    kernel_top_k: int,
) -> torch.Tensor:
    output = torch.full(
        (ROUTED_M, shape_n),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    layer.locks.zero_()
    ops.launch_kernel(
        configs=arm["configs"],
        inputs=inputs,
        outputs=output,
        input_scale=input_scale,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_scale_2=getattr(layer, "weight_scale_2", None),
        zero_point=None,
        bias=None,
        sorted_ids=metadata["sorted_ids"],
        expert_ids=metadata["expert_ids"],
        num_tokens_padded=metadata["num_tokens_padded"],
        expert_layout=None,
        locks=layer.locks,
        top_k=kernel_top_k,
        valid_shape_m=ROUTED_M,
    )
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise RuntimeError(f"arm {arm['arm']} output is not finite")
    return output


def _accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = (actual_fp32 - reference_fp32).abs()
    close = torch.isclose(actual_fp32, reference_fp32, rtol=RTOL, atol=ATOL)
    reference_l2 = torch.linalg.vector_norm(reference_fp32)
    normalized_l2 = torch.linalg.vector_norm(diff) / reference_l2
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "normalized_l2": float(normalized_l2.item()),
        "exact_fraction": float((actual == reference).float().mean().item()),
        "mismatch_fraction": float((~close).float().mean().item()),
        "rtol": RTOL,
        "atol": ATOL,
        "passed": bool(close.all().item()),
    }


def _public_arm_record(arm: dict) -> dict:
    return {key: value for key, value in arm.items() if key != "configs"}


@torch.inference_mode()
def main() -> None:
    args = _args()
    if args.w13_sk0_control and args.stage != "w13":
        raise ValueError("--w13-sk0-control requires --stage w13")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "H20" not in props.name or props.multi_processor_count != 78:
        raise RuntimeError(f"expected a 78-SM H20, got {props.name}/{props.multi_processor_count}")

    stage_config = STAGES[args.stage]
    shape_n = stage_config["shape_n"]
    shape_k = stage_config["shape_k"]
    historical_id = _historical_config_id(stage_config["tuning"])
    if historical_id != stage_config["historical_config_id"]:
        raise RuntimeError(f"exact tuning drift: {historical_id}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    layer = _new_layer(shape_n, shape_k)
    _fill_raw_parameters(layer)
    raw, raw_hashes = _snapshot_raw(layer)
    layer.transform()
    assert layer.humming_config is not None
    if not layer.humming_config.use_fused_e8m0_scale:
        raise RuntimeError("A/C resident layer did not select native fused mode-2 storage")
    resident = _resident_manifest(layer)

    arm_a = _compile_arm(
        layer,
        args.stage,
        "A-runtime-fused",
        fixed=False,
        w13_sk0_control=args.w13_sk0_control,
    )
    arm_c = _compile_arm(
        layer,
        args.stage,
        "C-fixed1-C-scale",
        fixed=True,
        w13_sk0_control=args.w13_sk0_control,
    )

    # Both arm records deliberately point at the exact same launch tensors.
    arm_resident = {"A": resident, "C": resident}
    if arm_resident["A"] != arm_resident["C"]:
        raise RuntimeError("A/C resident pointer or hash identity failed")
    compile_record = {
        "phase": "compile",
        "stage": args.stage,
        "historical_config_id": historical_id,
        "resident_identity": arm_resident,
        "kernels": {
            "A": _public_arm_record(arm_a),
            "C": _public_arm_record(arm_c),
        },
    }
    print(json.dumps(compile_record, indent=2), flush=True)
    _check_distinct_compiles(arm_a, arm_c)

    weight_ref = _raw_reference(layer, raw)
    del raw
    topk_ids, metadata, route_manifest = _routing(args.route_seed)
    inputs, input_scale, inputs_ref = _quantized_inputs(args.stage, shape_k, args.seed)
    reference = _fp32_reference(args.stage, inputs_ref, weight_ref, topk_ids, shape_n)

    outputs = {}
    repeat_bitwise = {}
    accuracy = {}
    for arm in (arm_a, arm_c):
        first = _run_arm(
            layer,
            arm,
            args.stage,
            inputs,
            input_scale,
            metadata,
            shape_n,
            stage_config["kernel_top_k"],
        )
        second = _run_arm(
            layer,
            arm,
            args.stage,
            inputs,
            input_scale,
            metadata,
            shape_n,
            stage_config["kernel_top_k"],
        )
        repeat_bitwise[arm["arm"]] = bool(torch.equal(first, second))
        if not repeat_bitwise[arm["arm"]]:
            raise RuntimeError(f"arm {arm['arm']} is not repeat-bitwise")
        accuracy[arm["arm"]] = _accuracy(first, reference)
        outputs[arm["arm"]] = first

    cross_accuracy = _accuracy(outputs[arm_c["arm"]], outputs[arm_a["arm"]])
    strict_mechanism_passed = (
        cross_accuracy["mismatch_fraction"] <= 1e-4 and cross_accuracy["normalized_l2"] <= 1e-3
    )
    bounded_stream_k_difference = (
        cross_accuracy["mismatch_fraction"] <= 0.03 and cross_accuracy["normalized_l2"] <= 0.003
    )
    is_w13 = args.stage == "w13"
    cross_accuracy["mechanism_passed"] = strict_mechanism_passed
    if strict_mechanism_passed:
        cross_accuracy["attribution"] = "direct"
    elif args.w13_sk0_control:
        cross_accuracy["attribution"] = "c-promotion-rounding"
    else:
        cross_accuracy["attribution"] = "c-promotion-rounding-plus-stream-k"
    if not strict_mechanism_passed and not (is_w13 and bounded_stream_k_difference):
        raise RuntimeError(f"A/C outputs disagree beyond the bounded gate: {cross_accuracy}")
    oracle_delta = {
        "mismatch_fraction": accuracy[arm_c["arm"]]["mismatch_fraction"]
        - accuracy[arm_a["arm"]]["mismatch_fraction"],
        "mean_abs": accuracy[arm_c["arm"]]["mean_abs"] - accuracy[arm_a["arm"]]["mean_abs"],
        "max_abs": accuracy[arm_c["arm"]]["max_abs"] - accuracy[arm_a["arm"]]["max_abs"],
    }
    # The load-time fused preprocessing is lossy for arbitrary synthetic raw
    # weights, so native A can have a tiny raw-oracle tail.  POC correctness is
    # gated on C agreeing with A and not adding a material oracle regression.
    oracle_regression_passed = oracle_delta["mismatch_fraction"] <= 1e-4 and oracle_delta["mean_abs"] <= max(
        1e-4, accuracy[arm_a["arm"]]["mean_abs"] * 0.1
    )
    if not oracle_regression_passed and not is_w13:
        raise RuntimeError(f"C adds raw-oracle mismatches beyond A: {oracle_delta}")
    oracle_delta["regression_passed"] = oracle_regression_passed

    document = {
        "schema_version": 1,
        "status": "pass" if strict_mechanism_passed and oracle_regression_passed else "conditional",
        "created_at": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "harness_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "environment": {
            "device": props.name,
            "sm_count": props.multi_processor_count,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "humming_module": humming.__file__,
        },
        "workload": {
            "stage": args.stage,
            "token_m": TOKEN_M,
            "routed_m": ROUTED_M,
            "shape_n": shape_n,
            "shape_k": shape_k,
            "num_experts": NUM_EXPERTS,
            "router_top_k": ROUTER_TOP_K,
            "kernel_top_k": stage_config["kernel_top_k"],
            "historical_config_id": historical_id,
            "schedule_variant": "w13-sk0-control" if args.w13_sk0_control else "exact-target",
            **route_manifest,
        },
        "raw_tensor_sha256": raw_hashes,
        "resident": resident,
        "resident_identity": {
            "same_object_contract": True,
            "A": resident,
            "C": resident,
        },
        "kernels": {
            "A": _public_arm_record(arm_a),
            "C": _public_arm_record(arm_c),
        },
        "correctness": {
            "repeat_bitwise": repeat_bitwise,
            "fp32_oracle": accuracy,
            "C_minus_A_oracle_error": oracle_delta,
            "C_vs_A": cross_accuracy,
        },
    }
    if args.output is not None:
        _atomic_write(args.output, document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
