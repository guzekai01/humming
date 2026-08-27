"""Contracts for the experimental mode-2 fixed-offset MXFP4 decoder."""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import math
import struct
from fractions import Fraction
from typing import ClassVar

import pytest
import torch

from humming.config import TuningConfig
from humming.jit.runtime import KernelRuntime

# The current fused LUT is intentionally tested at both offsets.  Offset zero
# is not a valid fixed-base decoder: E2M1 +/-0.5 underflows to signed zero.
_OFFSET0_FP8_BYTES = (
    0x00,
    0x00,
    0x08,
    0x0C,
    0x10,
    0x14,
    0x18,
    0x1C,
    0x80,
    0x80,
    0x88,
    0x8C,
    0x90,
    0x94,
    0x98,
    0x9C,
)
_OFFSET1_FP8_BYTES = (
    0x00,
    0x08,
    0x10,
    0x14,
    0x18,
    0x1C,
    0x20,
    0x24,
    0x80,
    0x88,
    0x90,
    0x94,
    0x98,
    0x9C,
    0xA0,
    0xA4,
)
_OFFSET0_CODEBOOK_SHA256 = "98bb6538541586c4380062441d7b07dbf85cb09cf4ede4de5c8fc29269fdc660"
_OFFSET1_CODEBOOK_SHA256 = "bb43f9a9793b426b32f3a7240c9757898568c252868b7a86d1e18ed76bb6fbb2"

# humming_pack_weight(mode=2) keeps magnitudes in natural order and permutes
# signs by this map.  The decoder reverses that representation into WGMMA's
# [row0-low, row1-low, row0-high, row1-high] register-word order.
_MODE2_SIGN_PERM = (0, 4, 1, 5, 2, 6, 3, 7)
_BASIS_INPUT_SHA256 = "434134510df7f491e8c4126a5d8c11fbee38d76a4f8586d71eebbaaebeb7917d"
_OFFSET0_BASIS_OUTPUT_SHA256 = "61539fd968b5e99f0aee4eb0bd5fc64439bb394cab4d67497bff5a6cd6ed630d"
_OFFSET1_BASIS_OUTPUT_SHA256 = "846e361aa2e17bd03622bd22abd6379ef6b864b3a110e8d3184d614fae449d26"
_RELATIVE_SCALE_E8M0_SHA256 = "1663455c3c04715ff62b3b4306824d0a6e6dcf1098a2687f46962547172ce26e"
_RELATIVE_SCALE_FP32_SHA256 = "bf7825b43dec81e2372dd2fce7b6b44f6dbc892ec3e1593d16cb193ee1d41ccf"

_E2M1_VALUES = (
    Fraction(0),
    Fraction(1, 2),
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(6),
    Fraction(0),
    Fraction(-1, 2),
    Fraction(-1),
    Fraction(-3, 2),
    Fraction(-2),
    Fraction(-3),
    Fraction(-4),
    Fraction(-6),
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode_e4m3_bits(value: int) -> Fraction:
    """Decode the finite E4M3 bit patterns used by this test exactly."""

    sign = -1 if value & 0x80 else 1
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        magnitude = Fraction(mantissa, 8) * Fraction(1, 2**6)
    else:
        magnitude = (1 + Fraction(mantissa, 8)) * Fraction(2) ** (exponent - 7)
    return sign * magnitude


def _dynamic_offset_byte(code: int, relative_exponent: int) -> int:
    """Expected current fused decoder byte for offsets in the valid [1, 12]."""

    assert 0 <= code < 16
    assert 1 <= relative_exponent <= 12
    magnitude = code & 0x7
    if magnitude == 0:
        value = 0
    else:
        value = _OFFSET1_FP8_BYTES[magnitude] + 8 * (relative_exponent - 1)
    return value | (0x80 if code & 0x8 else 0)


def _pack_mode2_row(codes: tuple[int, ...] | list[int]) -> int:
    assert len(codes) == 8
    result = 0
    for lane, code in enumerate(codes):
        magnitude = code & 0x7
        sign = codes[_MODE2_SIGN_PERM[lane]] & 0x8
        result |= (magnitude | sign) << (lane * 4)
    return result


def _fragment_output_bytes(
    row0: tuple[int, ...] | list[int],
    row1: tuple[int, ...] | list[int],
    codebook: tuple[int, ...],
) -> bytes:
    return bytes(
        (
            *(codebook[code] for code in row0[:4]),
            *(codebook[code] for code in row1[:4]),
            *(codebook[code] for code in row0[4:]),
            *(codebook[code] for code in row1[4:]),
        )
    )


def _decoder_basis() -> tuple[list[tuple[int, int]], bytes, bytes]:
    """Every E2M1 code in every logical lane of both mode-2 rows."""

    inputs = []
    offset0_outputs = bytearray()
    offset1_outputs = bytearray()
    for row_index in range(2):
        for lane in range(8):
            for code in range(16):
                rows = [[0] * 8, [0] * 8]
                rows[row_index][lane] = code
                inputs.append((_pack_mode2_row(rows[0]), _pack_mode2_row(rows[1])))
                offset0_outputs.extend(_fragment_output_bytes(rows[0], rows[1], _OFFSET0_FP8_BYTES))
                offset1_outputs.extend(_fragment_output_bytes(rows[0], rows[1], _OFFSET1_FP8_BYTES))
    return inputs, bytes(offset0_outputs), bytes(offset1_outputs)


def _u32_to_i32(value: int) -> int:
    return value if value < 2**31 else value - 2**32


def test_mxfp4_fused_lut_offset0_offset1_contract():
    assert _sha256(bytes(_OFFSET0_FP8_BYTES)) == _OFFSET0_CODEBOOK_SHA256
    assert _sha256(bytes(_OFFSET1_FP8_BYTES)) == _OFFSET1_CODEBOOK_SHA256

    # The literal offset-zero path loses +/-0.5.  Offset one is the smallest
    # current-LUT offset that keeps all 16 E2M1 codes representable.
    assert _OFFSET0_FP8_BYTES[0x1] == 0x00
    assert _OFFSET0_FP8_BYTES[0x9] == 0x80
    assert _OFFSET1_FP8_BYTES[0x1] == 0x08
    assert _OFFSET1_FP8_BYTES[0x9] == 0x88
    assert _decode_e4m3_bits(_OFFSET1_FP8_BYTES[0x1]) == Fraction(1, 64)
    assert _decode_e4m3_bits(_OFFSET1_FP8_BYTES[0x9]) == Fraction(-1, 64)

    for code, e2m1_value in enumerate(_E2M1_VALUES):
        assert _decode_e4m3_bits(_OFFSET1_FP8_BYTES[code]) == e2m1_value / 32
        if (code & 0x7) != 1:
            assert _decode_e4m3_bits(_OFFSET0_FP8_BYTES[code]) == e2m1_value / 64


def test_mxfp4_fixed1_late_scale_matches_runtime_fused_exhaustively():
    # Fused preprocessing stores a raw relative exponent r in [1, 12].  The
    # fixed-offset-one decoder must therefore apply exactly 2**(r-1) on C.
    canonical_e8m0_bytes = bytearray()
    fp32_scale_bits = bytearray()
    for relative_exponent in range(1, 13):
        canonical_e8m0_byte = relative_exponent + 126
        fp32_bits = canonical_e8m0_byte << 23
        canonical_e8m0_bytes.append(canonical_e8m0_byte)
        fp32_scale_bits.extend(struct.pack("<I", fp32_bits))
        fp32_scale = struct.unpack("<f", struct.pack("<I", fp32_bits))[0]
        assert fp32_scale == 2 ** (relative_exponent - 1)

        for code in range(16):
            runtime_byte = _dynamic_offset_byte(code, relative_exponent)
            runtime_value = _decode_e4m3_bits(runtime_byte)
            fixed_then_late = _decode_e4m3_bits(_OFFSET1_FP8_BYTES[code])
            fixed_then_late *= 2 ** (relative_exponent - 1)
            assert fixed_then_late == runtime_value

    assert _sha256(bytes(canonical_e8m0_bytes)) == _RELATIVE_SCALE_E8M0_SHA256
    assert _sha256(bytes(fp32_scale_bits)) == _RELATIVE_SCALE_FP32_SHA256


def test_mxfp4_mode2_register_order_marker_and_basis_hashes():
    row0 = (1, 2, 3, 4, 5, 6, 7, 8)
    row1 = (9, 10, 11, 12, 13, 14, 15, 0)
    assert _pack_mode2_row(row0) == 0x87654321
    assert _pack_mode2_row(row1) == 0x0FEDCBA9
    marker = _fragment_output_bytes(row0, row1, _OFFSET1_FP8_BYTES)
    assert struct.unpack("<4I", marker) == (
        0x18141008,
        0x98949088,
        0x8024201C,
        0x00A4A09C,
    )

    inputs, offset0_outputs, offset1_outputs = _decoder_basis()
    input_bytes = b"".join(struct.pack("<II", *values) for values in inputs)
    assert len(inputs) == 2 * 8 * 16
    assert _sha256(input_bytes) == _BASIS_INPUT_SHA256
    assert _sha256(offset0_outputs) == _OFFSET0_BASIS_OUTPUT_SHA256
    assert _sha256(offset1_outputs) == _OFFSET1_BASIS_OUTPUT_SHA256


def _tuning_config(*, fixed: bool) -> TuningConfig:
    # Set use_cp_async explicitly so this serialization test remains CPU-only.
    return TuningConfig(
        block_shape=(8, 128, 128),
        warp_shape=(8, 32, 128),
        use_mode2_fixed_mxfp4_c_scale=fixed,
        use_stream_k=False,
        num_stages=5,
        num_ctas_per_sm=4,
        use_cp_async=False,
    )


def test_mode2_fixed_flag_changes_python_and_cpp_kernel_keys():
    runtime_fused = _tuning_config(fixed=False)
    fixed_c_scale = _tuning_config(fixed=True)

    assert runtime_fused.to_str() != fixed_c_scale.to_str()
    assert '"use_mode2_fixed_mxfp4_c_scale": false' in runtime_fused.to_str()
    assert '"use_mode2_fixed_mxfp4_c_scale": true' in fixed_c_scale.to_str()

    runtime_cpp = runtime_fused.to_cpp_str(TuningConfig)
    fixed_cpp = fixed_c_scale.to_cpp_str(TuningConfig)
    assert runtime_cpp != fixed_cpp
    assert "static constexpr auto kUseMode2FixedMxfp4CScale = false;" in runtime_cpp
    assert "static constexpr auto kUseMode2FixedMxfp4CScale = true;" in fixed_cpp


_PROBE_CODE = r"""
#include <humming/datatype/dequant_fused.cuh>

__global__ void mxfp4_mode2_fixed_decoder_contract_probe(
    const uint32_t *inputs,
    uint32_t *offset0_outputs,
    uint32_t *offset1_outputs,
    uint32_t count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;

  const uint32_t *qb = inputs + index * 2;
  uint32_t offset0[4];
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint2 decoded = fused_dequant_single_for_mxfp4<Float8E4M3>(qb[i], 0);
    offset0[i * 2] = decoded.x;
    offset0[i * 2 + 1] = decoded.y;
  }
  uint32_t tmp = offset0[1];
  offset0[1] = offset0[2];
  offset0[2] = tmp;

  uint32_t offset1[4];
  fixed_dequant_for_mxfp4<Float8E4M3, 1, true, 1>(qb, offset1);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    offset0_outputs[index * 4 + i] = offset0[i];
    offset1_outputs[index * 4 + i] = offset1[i];
  }
}
"""


@dataclasses.dataclass(kw_only=True)
class _Mxfp4Mode2FixedDecoderContractProbe(KernelRuntime):
    name: ClassVar[str] = "mxfp4_mode2_fixed_decoder_contract_probe"
    disable_fast_math: ClassVar[bool] = True

    def init_kernel(self):
        self.code = _PROBE_CODE
        self.kernel_expr = self.name
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        offset0_outputs: torch.Tensor,
        offset1_outputs: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cbd

        assert inputs.dtype == torch.int32 and inputs.shape[1] == 2
        assert offset0_outputs.dtype == torch.int32
        assert offset1_outputs.dtype == torch.int32
        assert offset0_outputs.shape == offset1_outputs.shape == (inputs.shape[0], 4)

        self.check_context()
        count = inputs.shape[0]
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(count / 128)
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(inputs.device).cuda_stream
        arg_values = (
            inputs.data_ptr(),
            offset0_outputs.data_ptr(),
            offset1_outputs.data_ptr(),
            count,
        )
        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder probe")
def test_mxfp4_mode2_fixed_decoder_device_probe():
    basis_inputs, expected_offset0, expected_offset1 = _decoder_basis()
    inputs = torch.tensor(
        [[_u32_to_i32(value) for value in pair] for pair in basis_inputs],
        dtype=torch.int32,
        device="cuda",
    )
    offset0_outputs = torch.empty((len(basis_inputs), 4), dtype=torch.int32, device="cuda")
    offset1_outputs = torch.empty_like(offset0_outputs)

    _Mxfp4Mode2FixedDecoderContractProbe()(
        inputs,
        offset0_outputs,
        offset1_outputs,
    )
    torch.cuda.synchronize()

    actual_offset0 = offset0_outputs.cpu().contiguous().numpy().tobytes()
    actual_offset1 = offset1_outputs.cpu().contiguous().numpy().tobytes()
    assert actual_offset0 == expected_offset0
    assert actual_offset1 == expected_offset1
    assert _sha256(actual_offset0) == _OFFSET0_BASIS_OUTPUT_SHA256
    assert _sha256(actual_offset1) == _OFFSET1_BASIS_OUTPUT_SHA256
