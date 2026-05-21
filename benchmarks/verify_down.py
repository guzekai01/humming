"""Correctness gate for the humming MXFP4 W4A8 MoE down-gemm.

The tuner config (block/stage/cta/grid knobs) is performance-only: every config
computes the same GEMM, so the output must match a saved golden reference up to
floating-point accumulation-order noise. This script runs the down-gemm through
the current heuristic tuner and compares against the golden reference; it gates
on relative max error < 2% (rel_max is ~0 for tiling-invariant knobs such as
num_sms, and tiny if the K-reduction order changes). A broken config diverges
far past that bound.

  python benchmarks/verify_down.py --save-ref   # capture golden (run on baseline)
  python benchmarks/verify_down.py              # compare candidate vs golden
"""
import argparse
import json
import sys

import torch

from humming import dtypes, ops  # noqa: F401
from humming.config import GemmType
from humming.layer import HummingLayer
from humming.tune import get_heuristics_config
from humming.utils.test import generate_random_moe_tensors, random_fill_tensor

SEED = 20260521


def run_down(shape_n, shape_k, num_experts, top_k, shape_m):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
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

    torch.manual_seed(SEED)
    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    meta = layer.humming_metas[""]

    actual_shape_m = shape_m * top_k  # is_moe_down
    torch.manual_seed(SEED)
    inputs = torch.randn((actual_shape_m, shape_k), dtype=torch.bfloat16, device="cuda:0")
    inputs, input_scale = ops.quant_input(inputs, "float8e4m3", None, group_size=0)

    tuning_config = get_heuristics_config(meta=meta, gemm_type=GemmType.INDEXED)
    routed = shape_m * top_k
    block_size_config = None
    for min_m, max_m, cfg in tuning_config:
        if routed > min_m and routed <= max_m:
            block_size_config = cfg["block_shape"][0]
            break
    assert block_size_config is not None

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    _, expert_layout, sorted_ids, expert_ids, num_tokens_padded = generate_random_moe_tensors(
        shape_m=shape_m,
        num_experts=num_experts,
        top_k=top_k,
        gemm_type=GemmType.INDEXED,
        block_size_config=block_size_config,
    )

    out = layer(
        inputs=inputs,
        input_scale=input_scale,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        compute_config=json.dumps({"gemm_type": GemmType.INDEXED.value}),
        tuning_config=tuning_config,
        top_k=1,
    )
    torch.cuda.synchronize()
    return out.detach().float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape_n", type=int, default=4096)
    p.add_argument("--shape_k", type=int, default=256)
    p.add_argument("--num_experts", type=int, default=256)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--shape_m", type=int, default=4096)
    p.add_argument("--ref", type=str, default="/root/bench-out/verify_down_ref.pt")
    p.add_argument("--save-ref", action="store_true")
    args = p.parse_args()

    out = run_down(args.shape_n, args.shape_k, args.num_experts, args.top_k, args.shape_m)
    bad = bool(torch.isnan(out).any() or torch.isinf(out).any())
    print(
        f"output shape={tuple(out.shape)} mean={out.mean():.5f} std={out.std():.5f} "
        f"absmax={out.abs().max():.5f} nan_or_inf={bad}"
    )
    if bad:
        print("VERIFY FAIL: NaN/Inf in output")
        sys.exit(1)

    if args.save_ref:
        torch.save(out, args.ref)
        print(f"saved reference -> {args.ref}")
        return

    ref = torch.load(args.ref)
    if ref.shape != out.shape:
        print(f"VERIFY FAIL: shape {tuple(out.shape)} != ref {tuple(ref.shape)}")
        sys.exit(1)
    # compute in float64: the test tensors span ~1e9, fp32 reductions overflow/cancel
    out64 = out.double()
    ref64 = ref.double()
    diff = (out64 - ref64).abs()
    ref_scale = ref64.abs().max().item()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_max = max_abs / (ref_scale + 1e-30)
    cos = torch.nn.functional.cosine_similarity(out64.flatten(), ref64.flatten(), dim=0).item()
    print(f"vs ref: max_abs={max_abs:.4g} rel_max={rel_max:.3e} mean_abs={mean_abs:.4g} cos={cos:.8f}")
    # same GEMM math; config changes perturb only FP accumulation order (rel_max ~0 if
    # tiling-invariant, tiny if K-reduction order changes). A broken config -> large rel_max.
    if rel_max < 0.02:
        print("VERIFY PASS")
    else:
        print("VERIFY FAIL: output diverges from reference")
        sys.exit(1)


if __name__ == "__main__":
    main()
