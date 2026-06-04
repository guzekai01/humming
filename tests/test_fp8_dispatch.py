"""Non-fused MXFP4 W4A8 + grouped FP8 input scale (DeepEP FP8 dispatch support).

Validates that with use_fused_e8m0_scale=False the Humming kernel computes
  A = float8e4m3 (grouped input scale) @ B = float4e2m1 (E8M0 g32 weight scale)
correctly on H20, including the asymmetric input g128 / weight g32 case needed for
DeepEP low-latency FP8 1x128 dispatch. A 16x per-group input-scale factor is injected
(and matched into the reference) so a dropped/permuted group scale cannot hide.

f16 accumulation is intentionally excluded: non-fused E8M0-on-C needs E8M0->FP16,
which hits the fp_to_fp exponent-bits static_assert (E8M0=8 exp bits, FP16=5);
E8M0->BF16 is special-cased so FP32 accumulation (c=bf16/fp16) is fine.
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


@pytest.mark.parametrize("mma_type", ["wgmma", "mma"])
@pytest.mark.parametrize("c_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "input_scale_group_size, weight_scale_group_size",
    [(0, 32), (128, 128), (128, 32)],
)
def test_fp8_dispatch(mma_type, c_dtype, input_scale_group_size, weight_scale_group_size):
    skip_if_unsupported(a_dtype="float8e4m3", mma_type=mma_type)
    a_dtype = dtypes.float8e4m3
    b_dtype = dtypes.float4e2m1
    bs_dtype = dtypes.float8e8m0
    c_dtype = dtypes.DataType.from_str(c_dtype)
    n = k = 1024
    m = 128

    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=n, k=k, group_size=weight_scale_group_size, dtype=b_dtype, scale_dtype=bs_dtype,
    )

    # Weight-scale layout flag by convention (test_scale.py:96-102), not hardcoded.
    if mma_type == "wgmma":
        to_apply_on_c = weight_scale_group_size == 0
    else:
        to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16
    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=(mma_type == "wgmma"))
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m, k=k, group_size=input_scale_group_size, dtype=a_dtype,
    )
    if input_scale_group_size > 0:
        ng = input_scale.shape[1]
        pattern = (2.0 ** ((torch.arange(ng, device=input_scale.device) % 5) - 2)).float()
        input_scale = input_scale * pattern
        inputs_ref = inputs_ref * pattern.repeat_interleave(input_scale_group_size)

    if mma_type == "wgmma":
        block_shape, warp_shape = (64, 128, 128), (64, 16, 128)
    else:
        block_shape, warp_shape = (16, 128, 64), (16, 32, 64)

    kernel = HummingKernel(
        shape_n=n,
        shape_k=k,
        block_shape=block_shape,
        warp_shape=warp_shape,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group",
        use_fused_e8m0_scale=False,
        use_f16_accum=False,
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((m, n), dtype=torch_dtype, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.5)


@pytest.mark.parametrize("input_scale_group_size, expect_fused", [(0, True), (128, False)])
def test_fp8_dispatch_auto_fuse_toggle(input_scale_group_size, expect_fused):
    """MXFP4 W4A8 auto-selects fused E8M0 only for per-token input (group 0).

    Grouped FP8 input (DeepEP 1x128 dispatch) must fall to non-fused so the grouped
    input scale is applied; that path uses GROUP weight scale (no per-tensor global),
    whereas the fused per-token path uses GROUP_TENSOR.
    """
    from humming.layer import HummingLayerMeta

    meta = HummingLayerMeta(
        shape_n=1024,
        shape_k=1024,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
        input_scale_group_size=input_scale_group_size,
        mma_type="wgmma",
    )
    assert meta.use_fused_e8m0_scale == expect_fused
    assert meta.is_group_weight_scale is True
    # GROUP_TENSOR (per-tensor global) only on the fused per-token path.
    assert meta.is_tensor_weight_scale == expect_fused
