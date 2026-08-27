#!/usr/bin/env python3
"""Paired K3 M32 timing for native A, fixed1+C C, and mode-3 D."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path

import torch
import triton
from k3_mode2_fixed1_m32 import (
    NUM_EXPERTS,
    ROUTED_M,
    STAGES,
    _accuracy,
    _atomic_write,
    _compile_arm,
    _fill_raw_parameters,
    _fp32_reference,
    _new_layer,
    _public_arm_record,
    _quantized_inputs,
    _raw_reference,
    _routing,
    _run_arm,
    _snapshot_raw,
)

from humming import ops
from humming.config import WeightScale2Type
from humming.transform import prepare_layer_config, transform_humming_tensors


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--route-seed", type=int, default=20260819)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--warmup-ms", type=float, default=5.0)
    parser.add_argument("--rep-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _build_explicit_layer(raw: dict[str, torch.Tensor], shape_n: int, shape_k: int):
    layer = _new_layer(shape_n, shape_k)
    config = prepare_layer_config(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=layer.weight_schema,
        input_schema=layer.input_schema,
        num_experts=NUM_EXPERTS,
        torch_dtype=torch.bfloat16,
    )
    config = dataclasses.replace(
        config,
        use_fused_e8m0_scale=False,
        weight_scale_2_type=WeightScale2Type.NONE,
    )
    tensors = transform_humming_tensors(
        config,
        {name: value.clone() for name, value in raw.items()},
    )
    for name, value in tensors.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
    layer.humming_config = config
    return layer


def _make_launch(layer, arm: dict, inputs, input_scale, metadata, shape_n: int, top_k: int):
    output = torch.empty((ROUTED_M, shape_n), dtype=torch.bfloat16, device="cuda:0")
    layer.locks.zero_()

    def launch():
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
            top_k=top_k,
            valid_shape_m=ROUTED_M,
        )

    return launch


@torch.inference_mode()
def main() -> None:
    args = _args()
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "H20" not in props.name or props.multi_processor_count != 78:
        raise RuntimeError(f"expected a 78-SM H20, got {props.name}/{props.multi_processor_count}")
    if args.blocks < 3:
        raise ValueError("--blocks must be at least 3")

    stage_config = STAGES[args.stage]
    shape_n = stage_config["shape_n"]
    shape_k = stage_config["shape_k"]
    kernel_top_k = stage_config["kernel_top_k"]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    fused_layer = _new_layer(shape_n, shape_k)
    _fill_raw_parameters(fused_layer)
    raw, raw_hashes = _snapshot_raw(fused_layer)
    explicit_layer = _build_explicit_layer(raw, shape_n, shape_k)
    fused_layer.transform()

    arm_a = _compile_arm(
        fused_layer,
        args.stage,
        "A-runtime-fused",
        fixed=False,
        w13_sk0_control=False,
    )
    arm_c = _compile_arm(
        fused_layer,
        args.stage,
        "C-fixed1-C-scale",
        fixed=True,
        w13_sk0_control=False,
    )
    arm_d = _compile_arm(
        explicit_layer,
        args.stage,
        "D-mode3-explicit",
        fixed=False,
        w13_sk0_control=False,
    )

    topk_ids, metadata, route_manifest = _routing(args.route_seed)
    inputs, input_scale, inputs_ref = _quantized_inputs(args.stage, shape_k, args.seed)
    weight_ref = _raw_reference(fused_layer, raw)
    reference = _fp32_reference(args.stage, inputs_ref, weight_ref, topk_ids, shape_n)
    del raw, weight_ref

    layers = {"A": fused_layer, "C": fused_layer, "D": explicit_layer}
    arms = {"A": arm_a, "C": arm_c, "D": arm_d}
    correctness = {}
    correctness_outputs = {}
    for name in ("A", "C", "D"):
        output = _run_arm(
            layers[name],
            arms[name],
            args.stage,
            inputs,
            input_scale,
            metadata,
            shape_n,
            kernel_top_k,
        )
        correctness[name] = _accuracy(output, reference)
        correctness_outputs[name] = output
    correctness["C_vs_A"] = _accuracy(correctness_outputs["C"], correctness_outputs["A"])
    correctness["C_vs_D"] = _accuracy(correctness_outputs["C"], correctness_outputs["D"])

    launches = {
        name: _make_launch(
            layers[name],
            arms[name],
            inputs,
            input_scale,
            metadata,
            shape_n,
            kernel_top_k,
        )
        for name in ("A", "C", "D")
    }
    orders = (
        ("A", "C", "D"),
        ("D", "C", "A"),
        ("C", "A", "D"),
        ("D", "A", "C"),
        ("C", "D", "A"),
        ("A", "D", "C"),
    )
    samples = {name: [] for name in launches}
    sequence = []
    for block in range(args.blocks):
        order = orders[block % len(orders)]
        for name in order:
            value = float(
                triton.testing.do_bench(
                    launches[name],
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                )
            )
            samples[name].append(value)
            sequence.append({"block": block, "arm": name, "ms": value})

    medians = {name: statistics.median(values) for name, values in samples.items()}
    document = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "command": __import__("sys").argv,
        "environment": {
            "device": props.name,
            "sm_count": props.multi_processor_count,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_clock_lock_mhz": os.environ.get("BENCH_GPU_CLOCK_MHZ"),
        },
        "workload": {
            "stage": args.stage,
            "token_m": 32,
            "routed_m": ROUTED_M,
            "shape_n": shape_n,
            "shape_k": shape_k,
            **route_manifest,
        },
        "raw_tensor_sha256": raw_hashes,
        "kernels": {name: _public_arm_record(arm) for name, arm in arms.items()},
        "correctness": correctness,
        "timing": {
            "protocol": "rotating paired blocks; triton do_bench default L2 flush",
            "blocks": args.blocks,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "samples_ms": samples,
            "sequence": sequence,
            "median_ms": medians,
            "C_vs_A_speedup": medians["A"] / medians["C"],
            "C_vs_D_speedup": medians["D"] / medians["C"],
            "D_vs_A_speedup": medians["A"] / medians["D"],
        },
    }
    _atomic_write(args.output, document)
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
