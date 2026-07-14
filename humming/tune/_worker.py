import json
import math
import os
from types import SimpleNamespace


def _format_short_error(exc):
    return f"{type(exc).__name__}: {exc}"[:500]


def create_layer(args, gemm_type):
    from humming import dtypes
    from humming.layer import HummingLayer
    from humming.utils.test import random_fill_tensor

    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(args.c_dtype)]
    layer = HummingLayer(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        num_experts=args.num_experts if gemm_type.value != "dense" else 0,
        weight_config={
            "dtype": args.b_dtype,
            "group_size": args.weight_scale_group_size,
            "scale_dtype": args.bs_dtype,
            "has_zero_point": False,
            "is_fp_zero_point": False,
        },
        input_config={"dtype": args.a_dtype, "group_size": args.input_scale_group_size},
        torch_dtype=torch_dtype,
    ).to("cuda:0")

    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    return layer, torch_dtype


def default_expert_max_tokens(args, shape_m):
    if args.expert_max_tokens is not None:
        return args.expert_max_tokens
    tokens_per_expert = math.ceil(shape_m * args.top_k / args.num_experts / 64) * 64
    multiplier = 1 if args.balanced else 2
    return max(64, tokens_per_expert * multiplier)


def indexed_input_rows(shape_m, top_k, is_moe_down):
    """Activation rows and routing token count for indexed gemm.

    ``shape_m`` is routed rows (tokens * top_k), matching the serving bucket key
    ``estimate_local_valid_shape_m() = topk_ids.nelement()``. w13 consumes
    per-token activations (``shape_m // top_k`` rows); w2 consumes the routed
    rows themselves (``shape_m`` rows). Routing metadata is built from the token
    count for both.
    """
    if shape_m % top_k != 0:
        raise ValueError(f"indexed shape_m={shape_m} must be divisible by top_k={top_k}")
    routing_tokens = shape_m // top_k
    activation_rows = shape_m if is_moe_down else routing_tokens
    return activation_rows, routing_tokens


def make_inputs(args, gemm_type, torch_dtype, shape_m, block_size_config=None):
    import torch

    from humming import ops
    from humming.utils.test import generate_random_moe_tensors

    # Number of token rows fed to the routing generator (topk_ids is
    # [routing_tokens, top_k]). For indexed this is derived from routed rows;
    # for masked it stays the token count consumed by default_expert_max_tokens.
    routing_tokens = shape_m
    if gemm_type.value == "dense":
        actual_shape_m = shape_m
        expert_max_tokens = None
    elif gemm_type.value == "indexed":
        actual_shape_m, routing_tokens = indexed_input_rows(
            shape_m, args.top_k, args.is_moe_down
        )
        expert_max_tokens = None
    else:
        expert_max_tokens = default_expert_max_tokens(args, shape_m)
        actual_shape_m = args.num_experts * expert_max_tokens

    # Metadata is cached per block_m; keep activations identical across configs.
    torch.cuda.manual_seed(shape_m)
    inputs = torch.randn((actual_shape_m, args.shape_k), dtype=torch_dtype, device="cuda:0")
    input_scale = None
    if args.a_dtype not in ["float16", "bfloat16"]:
        inputs, input_scale = ops.quant_input(
            inputs,
            args.a_dtype,
            group_size=args.input_scale_group_size,
        )

    torch.cuda.manual_seed(shape_m)
    if gemm_type.value == "dense":
        expert_layout = None
        sorted_ids = None
        expert_ids = None
        num_tokens_padded = None
    else:
        moe_tensors = generate_random_moe_tensors(
            shape_m=routing_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            gemm_type=gemm_type,
            balanced=args.balanced,
            block_size_config=block_size_config,
            expert_max_tokens=expert_max_tokens,
        )
        _, expert_layout, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

    return {
        "inputs": inputs,
        "input_scale": input_scale,
        "expert_layout": expert_layout,
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_tokens_padded": num_tokens_padded,
        "actual_shape_m": actual_shape_m,
        "expert_max_tokens": expert_max_tokens,
    }


def make_run(layer, batch, compute_config_json, tuning_config, gemm_type, shape_m, top_k):
    def run():
        valid_shape_m = 0
        if gemm_type.value == "grouped_masked":
            valid_shape_m = shape_m * top_k
        return layer(
            inputs=batch["inputs"],  # noqa: B026
            input_scale=batch["input_scale"],  # noqa: B026
            sorted_ids=batch["sorted_ids"],  # noqa: B026
            expert_ids=batch["expert_ids"],  # noqa: B026
            num_tokens_padded=batch["num_tokens_padded"],  # noqa: B026
            expert_layout=batch["expert_layout"],  # noqa: B026
            valid_shape_m=valid_shape_m,
            compute_config=compute_config_json,
            tuning_config=tuning_config,  # noqa: B026
            top_k=top_k,
        )

    return run


def masked_valid_rows(expert_layout, expert_max_tokens):
    import torch

    rows = []
    for expert_id, count in enumerate(expert_layout.cpu().tolist()):
        start = expert_id * expert_max_tokens
        rows.extend(range(start, start + count))
    return torch.tensor(rows, dtype=torch.long, device="cuda:0")


def comparable_output(output, compare_rows):
    output = output.view(-1, output.size(-1))
    if compare_rows is None:
        return output
    return output.index_select(0, compare_rows)


def worker_main(args_dict, cmd_q, res_q):
    import torch
    import triton

    from humming.config.enum import GemmType

    args = SimpleNamespace(**args_dict)
    gemm_type = GemmType(args.gemm_type)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    layer, torch_dtype = create_layer(args, gemm_type)
    compute_config_json = json.dumps({"use_f16_accum": False, "gemm_type": gemm_type.value})
    batches = {}
    refs = {}

    layer_top_k = args.top_k if not getattr(args, "is_moe_down", False) else 1

    def get_batch(shape_m, block_size_config=None):
        key = (shape_m, block_size_config)
        if key not in batches:
            if not any(batch_key[0] == shape_m for batch_key in batches):
                batches.clear()
            batches[key] = make_inputs(args, gemm_type, torch_dtype, shape_m, block_size_config)
        return batches[key]

    res_q.put({"op": "ready"})
    while True:
        msg = cmd_q.get()
        op = msg["op"]
        if op == "stop":
            os._exit(0)
        shape_m = msg["m"]
        try:
            block_size_config = None
            if gemm_type.value == "indexed":
                block_size_config = msg["config"]["block_shape"][0]
            batch = get_batch(shape_m, block_size_config)
            run = make_run(
                layer,
                batch,
                compute_config_json,
                msg["config"],
                gemm_type,
                shape_m,
                layer_top_k,
            )
            if op == "bench":
                ms = triton.testing.do_bench(run, warmup=msg["warmup"], rep=msg["rep"])
                res_q.put({"op": op, "ms": ms, "error": ""})
            elif op == "set_ref":
                output = run()
                torch.cuda.synchronize()
                rows = None
                if gemm_type.value == "grouped_masked":
                    rows = masked_valid_rows(batch["expert_layout"], batch["expert_max_tokens"])
                refs.clear()
                refs[shape_m] = (comparable_output(output, rows).clone(), rows)
                res_q.put({"op": op, "error": ""})
            elif op == "compare":
                ref_output, rows = refs[shape_m]
                output = run()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    comparable_output(output, rows), ref_output, rtol=0.05, atol=0.1
                )
                res_q.put({"op": op, "passed": True, "error": ""})
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                os._exit(43)
            if op == "compare":
                res_q.put({"op": op, "passed": False, "error": _format_short_error(exc)})
            else:
                res_q.put({"op": op, "ms": float("inf"), "error": _format_short_error(exc)})
