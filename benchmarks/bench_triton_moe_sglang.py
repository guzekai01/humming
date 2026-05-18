"""Triton fused-MoE bench using sglang (not vllm) as the kernel provider.

Differences from benchmarks/bench_triton_moe.py:
  * Imports `invoke_fused_moe_kernel`, `get_moe_configs`, `get_default_config`
    from sglang.srt.layers.moe.moe_runner.triton_utils.
  * sglang's invoke expects unquantized A (bf16/fp16) and quantizes it
    internally for fp8_w8a8/int8_w8a8 paths. The earlier vllm-based bench
    pre-quantized A outside the timed region. To keep this script honest the
    a_dtype is fixed to bf16/fp16 (matches sglang's API), and timing includes
    sglang's internal activation quant kernel.
  * sglang's invoke takes additional positional args (bias, B_zp, topk_ids),
    a mandatory topk_weights tensor (even when mul_routed_weight=False), and
    extra kwargs (is_marlin etc. in get_default_config).
"""

import argparse
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from tqdm import tqdm

# sglang reads global server args inside MoE config / kernel paths; stub it
# before any sglang module pulls it, to avoid `Global server args is not set yet!`.
import sglang.srt.server_args as _sa
_sa._global_server_args = SimpleNamespace(
    enable_deterministic_inference=False,
    enable_torch_compile=False,
    moe_runner_backend="triton",
)

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_default_config,
    get_moe_configs,
)

from humming.utils.test import generate_random_moe_tensors, save_benchmark_result


def get_triton_moe_config(num_experts, shape_n, shape_k, shape_m, top_k,
                          is_moe_down, block_shape, weight_torch_dtype):
    if weight_torch_dtype == torch.float8_e4m3fn:
        dtype = "fp8_w8a8"
    elif weight_torch_dtype == torch.int8:
        dtype = "int8_w8a8"
    else:
        dtype = None

    if not is_moe_down:
        shape_n = shape_n // 2
    else:
        shape_n, shape_k = shape_k, shape_n

    # sglang stores up-proj configs as `..._N=...,block_shape=...json` and
    # down-proj configs as `..._down.json`. Pass `down_moe` so it picks the
    # right suffix; if no _down file exists sglang returns None and we fall
    # back to get_default_config (NOT silently to the up-proj tuning).
    if block_shape is None:
        configs = get_moe_configs(num_experts, shape_n, dtype, down_moe=is_moe_down)
    else:
        configs = get_moe_configs(num_experts, shape_n, dtype,
                                   block_shape[0], block_shape[1],
                                   down_moe=is_moe_down)
    if configs is None and block_shape is None:
        configs = get_moe_configs(num_experts, shape_n, dtype, 128, 128,
                                   down_moe=is_moe_down)
    if configs is not None:
        cfg = dict(configs[min(configs.keys(), key=lambda x: abs(x - shape_m))])
        # _down configs (newer tuning) carry USE_TMA / similar keys that this
        # sglang's fused_moe_kernel doesn't declare; strip them so triton's
        # **kwargs unpack doesn't KeyError. Core tuning (BLOCK_SIZE_M/N/K,
        # GROUP_SIZE_M, num_warps, num_stages) is preserved.
        for unknown in ("USE_TMA",):
            cfg.pop(unknown, None)
        return cfg
    return get_default_config(shape_m, num_experts, shape_n, shape_k, top_k, dtype,
                              is_marlin=False, block_shape=block_shape)


def bench_triton_moe(shape_n, shape_k, num_experts, top_k, is_moe_down,
                     weight_dtype, act_dtype, out_dtype,
                     block_shape=None, balanced=False, shape_m_list=None):
    if isinstance(block_shape, str):
        block_shape = [int(x) for x in block_shape.split("x")]
    weight_torch_dtype = {"int8": torch.int8, "float8e4m3": torch.float8_e4m3fn,
                          "float16": torch.float16, "bfloat16": torch.bfloat16}[weight_dtype]
    act_torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[act_dtype]

    if weight_torch_dtype in (torch.int8, torch.float8_e4m3fn):
        weight = torch.randint(-120, 120, (num_experts, shape_n, shape_k),
                                dtype=torch.int8, device="cuda:0").view(weight_torch_dtype)
    else:
        weight = torch.randn((num_experts, shape_n, shape_k), dtype=weight_torch_dtype, device="cuda:0")

    if weight_torch_dtype.itemsize == 2:
        weight_scale = None
    elif block_shape is None:
        weight_scale = torch.randn((num_experts, shape_n), dtype=torch.float32, device="cuda:0")
    else:
        weight_scale = torch.randn((num_experts, shape_n // block_shape[0], shape_k // block_shape[1]),
                                    dtype=torch.float32, device="cuda:0")

    default_shape_m_list = [2 ** i for i in range(15)]
    benchmark_result = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        if is_moe_down:
            inputs = torch.randn((shape_m * top_k, shape_k), dtype=act_torch_dtype, device="cuda:0")
        else:
            inputs = torch.randn((shape_m, shape_k), dtype=act_torch_dtype, device="cuda:0")

        torch.cuda.manual_seed(shape_m)
        config = get_triton_moe_config(num_experts, shape_n, shape_k, shape_m, top_k,
                                        is_moe_down, block_shape, weight_torch_dtype)
        topk_ids_full, _, sorted_ids, expert_ids, num_tokens_padded = generate_random_moe_tensors(
            shape_m=shape_m, num_experts=num_experts, top_k=top_k,
            balanced=balanced, block_size_config=config["BLOCK_SIZE_M"],
        )
        topk_weights = torch.ones((shape_m, top_k), dtype=torch.float32, device="cuda:0")

        def run():
            outputs = torch.empty((shape_m, top_k, shape_n), dtype=act_torch_dtype, device="cuda:0")
            invoke_fused_moe_kernel(
                A=inputs, B=weight, bias=None, C=outputs,
                A_scale=None, B_scale=weight_scale, B_zp=None,
                topk_weights=topk_weights, topk_ids=topk_ids_full,
                sorted_token_ids=sorted_ids, expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_padded,
                mul_routed_weight=False,
                top_k=1 if is_moe_down else top_k,
                config=config,
                compute_type=getattr(tl, out_dtype),
                use_fp8_w8a8=(weight_dtype == "float8e4m3"),
                use_int8_w8a8=(weight_dtype == "int8"),
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=(block_shape is None),
                block_shape=block_shape,
            )
            return outputs

        torch.cuda.synchronize()
        outputs = run()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        num_actived = len(set(expert_ids.tolist()))
        nbytes = inputs.nbytes + outputs.nbytes
        nbytes += weight.nbytes // num_experts * num_actived
        if weight_scale is not None:
            nbytes += weight_scale.nbytes // num_experts * num_actived

        benchmark_result.append({
            "shape_m": shape_m, "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * top_k * 2 / t / 1e9,
            "config": config,
        })
    return benchmark_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape_n", type=int, required=True)
    p.add_argument("--shape_k", type=int, required=True)
    p.add_argument("--weight_dtype", type=str, choices=["int8", "float8e4m3"], required=True)
    p.add_argument("--act_dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--out_dtype", type=str, choices=["float16", "bfloat16"], required=True)
    p.add_argument("--num_experts", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--is_moe_down", action="store_true")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--shape_m_list", type=int, nargs="+", default=None)
    p.add_argument("--block_shape", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    args = p.parse_args()

    res = bench_triton_moe(
        shape_n=args.shape_n, shape_k=args.shape_k,
        num_experts=args.num_experts, top_k=args.top_k,
        is_moe_down=args.is_moe_down,
        weight_dtype=args.weight_dtype, act_dtype=args.act_dtype, out_dtype=args.out_dtype,
        block_shape=args.block_shape, balanced=args.balanced,
        shape_m_list=args.shape_m_list,
    )
    if args.output_file:
        # save_benchmark_result reads `a_dtype` for the tops_bench peak-TOPS probe.
        args.a_dtype = args.weight_dtype
        save_benchmark_result(res, args, packages=["sglang", "triton"])
    print("\nshape_m | time(ms) | mem GB/s | TFLOPS")
    for r in res:
        print(f"{r['shape_m']:>7} | {r['time']*1000:>8.4f} | {r['memory_gbps']:>8.1f} | {r['compute_tops']:>8.2f}")


if __name__ == "__main__":
    main()
