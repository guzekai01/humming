"""Non-fused MXFP4 W4A8 + grouped FP8 input scale in the grouped-masked MoE shape.

Production decode GEMM counterpart to test_fp8_dispatch: per-expert E8M0 weight scale,
a deterministic expert layout covering masking boundaries, and an exact-zero check on
padding rows. See test_fp8_dispatch for the f16-accum exclusion rationale.
"""

import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import (
    generate_random_inputs,
    generate_random_weight,
    skip_if_unsupported,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)

NUM_EXPERTS = 8
EXPERT_MAX_TOKENS = 256
BLOCK_M = 64
# per-expert token counts (each <= EXPERT_MAX_TOKENS) covering masking boundaries vs
# BLOCK_M=64: zero / single / sub-block / exact-block / block+1 / multi-block+1 /
# partial-last-block / full.
EXPERT_LAYOUT_VALS = [0, 1, 63, 64, 65, 129, 200, EXPERT_MAX_TOKENS]


@pytest.mark.parametrize("mma_type", ["wgmma", "mma"])
@pytest.mark.parametrize("c_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "input_scale_group_size, weight_scale_group_size", [(0, 32), (128, 32)]
)
def test_fp8_dispatch_moe(mma_type, c_dtype, input_scale_group_size, weight_scale_group_size):
    skip_if_unsupported(a_dtype="float8e4m3", mma_type=mma_type)
    a_dtype = dtypes.float8e4m3
    b_dtype = dtypes.float4e2m1
    bs_dtype = dtypes.float8e8m0
    c_dtype = dtypes.DataType.from_str(c_dtype)
    n = k = 1024

    assert len(EXPERT_LAYOUT_VALS) == NUM_EXPERTS
    assert max(EXPERT_LAYOUT_VALS) <= EXPERT_MAX_TOKENS
    expert_layout = torch.tensor(EXPERT_LAYOUT_VALS, dtype=torch.int32, device="cuda:0")

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

    m_new = NUM_EXPERTS * EXPERT_MAX_TOKENS
    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m_new, k=k, group_size=input_scale_group_size, dtype=a_dtype,
    )
    if input_scale_group_size > 0:
        ng = input_scale.shape[1]
        pattern = (2.0 ** ((torch.arange(ng, device=input_scale.device) % 5) - 2)).float()
        input_scale = input_scale * pattern
        inputs_ref = inputs_ref * pattern.repeat_interleave(input_scale_group_size)

    if mma_type == "wgmma":
        block_shape, warp_shape = (BLOCK_M, 128, 128), (BLOCK_M, 16, 128)
    else:
        block_shape, warp_shape = (BLOCK_M, 128, 64), (BLOCK_M, 32, 64)

    kernel = HummingKernel(
        shape_n=n,
        shape_k=k,
        block_shape=block_shape,
        warp_shape=warp_shape,
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
        gemm_type="grouped_masked",
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((m_new, n), dtype=torch_dtype, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        expert_layout=expert_layout,
    ).view(-1, n)

    outputs_ref = torch.zeros_like(outputs)
    active_mask = torch.zeros(m_new, dtype=torch.bool, device=outputs.device)
    for e in range(NUM_EXPERTS):
        o1 = EXPERT_MAX_TOKENS * e
        o2 = o1 + int(expert_layout[e])
        if o2 > o1:
            active_mask[o1:o2] = True
            outputs_ref[o1:o2] = inputs_ref[o1:o2].matmul(weight_ref[e].T).to(torch_dtype)

    padding_mask = ~active_mask
    assert (outputs[padding_mask] == 0).all(), "padding rows must be exactly zero"
    torch.testing.assert_close(
        outputs[active_mask], outputs_ref[active_mask], rtol=0.05, atol=0.5
    )
