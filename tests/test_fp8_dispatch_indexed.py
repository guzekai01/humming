"""Non-fused MXFP4 W4A8 + grouped FP8 input scale in the INDEXED MoE shape.

This is the prefill path under FP8 dispatch (DeepEP normal -> indexed gemm). Neither
existing test covers the combination: test_indexed_gemm_input_scale is indexed +
group-128 input but uint4 weight; test_fp8_dispatch / _moe cover MXFP4 + group input
but only dense / grouped_masked. This closes the indexed + MXFP4(E8M0) + group-128 gap.

  A  = float8e4m3, input scale group=128
  B  = float4e2m1, weight scale = E8M0 group=32 (MXFP4), per-expert
  use_fused_e8m0_scale = False, weight_scale_type = "group", gemm_type = "indexed"

Conventions match test_fp8_dispatch{,_moe}: to_apply_on_c by convention, no f16 accum
(E8M0-on-C unsupported), element-wise allclose, 16x input-scale injection. Differential:
ctrl (input g0 / weight g32) vs target (input g128 / weight g32).

Run on H20 (sm90). Usage: python -m pytest tests/test_fp8_dispatch_indexed.py -v
"""

import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
    skip_if_unsupported,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)

ATOL, RTOL = 0.5, 0.05

NUM_EXPERTS = 8
TOP_K = 1
M = 256
BLOCK_M = 48  # must match the block_size_config given to generate_random_moe_tensors


@pytest.mark.parametrize("mma_type", ["wgmma", "mma"])
@pytest.mark.parametrize("c_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "input_scale_group_size, weight_scale_group_size", [(0, 32), (128, 32)]
)
def test_fp8_dispatch_indexed(mma_type, c_dtype, input_scale_group_size, weight_scale_group_size):
    skip_if_unsupported(a_dtype="float8e4m3", mma_type=mma_type)
    a_dtype = dtypes.float8e4m3
    b_dtype = dtypes.float4e2m1
    bs_dtype = dtypes.float8e8m0
    c_dtype = dtypes.DataType.from_str(c_dtype)
    n = k = 1024

    topk_ids, _, sorted_token_ids, expert_ids, num_tokens_padded = generate_random_moe_tensors(
        M, num_experts=NUM_EXPERTS, top_k=TOP_K, block_size_config=BLOCK_M,
    )

    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=n, k=k, group_size=weight_scale_group_size, dtype=b_dtype,
        scale_dtype=bs_dtype, num_experts=NUM_EXPERTS,
    )

    if mma_type == "wgmma":
        to_apply_on_c = weight_scale_group_size == 0
    else:
        to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16
    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=(mma_type == "wgmma"))
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=M, k=k, group_size=input_scale_group_size, dtype=a_dtype,
    )
    if input_scale_group_size > 0:
        ng = input_scale.shape[1]
        pattern = (2.0 ** ((torch.arange(ng, device=input_scale.device) % 5) - 2)).float()
        input_scale = input_scale * pattern
        inputs_ref = inputs_ref * pattern.repeat_interleave(input_scale_group_size)

    a_bits = a_dtype.num_bits
    kernel = HummingKernel(
        shape_n=n,
        shape_k=k,
        block_shape=(BLOCK_M, a_bits * 16, 512 // a_bits),
        warp_shape=(BLOCK_M, a_bits * 4, 512 // a_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_experts=NUM_EXPERTS,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group",
        use_fused_e8m0_scale=False,
        use_f16_accum=False,
        has_bias=False,
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
        use_stream_k=False,
        gemm_type="indexed",
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.empty((M * TOP_K, n), dtype=torch_dtype, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        sorted_ids=sorted_token_ids,
        top_k=TOP_K,
    ).view(-1, n)

    outputs_ref = torch.empty_like(outputs)
    for expert_id in range(NUM_EXPERTS):
        outputs_index = torch.where(topk_ids.view(-1) == expert_id)[0]
        inputs_index = outputs_index // TOP_K
        if inputs_index.size(0):
            outputs_ref[outputs_index] = (
                inputs_ref[inputs_index].matmul(weight_ref[expert_id].T).to(torch_dtype)
            )

    torch.testing.assert_close(outputs, outputs_ref, rtol=RTOL, atol=ATOL)
