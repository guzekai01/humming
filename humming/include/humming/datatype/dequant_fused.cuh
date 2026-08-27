#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class TargetType>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4(const uint32_t qb, const uint32_t exp_offset) {
  static_assert(std::is_same<TargetType, Float8E4M3>::value || std::is_same<TargetType, Int8>::value);
  return {0, 0};
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Float8E4M3>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t exp_offset_buffer1 = (exp_offset * 0x08080800) + ((0x03020100 << 2) - 0x00000400);
  uint32_t exp_offset_buffer2 = (exp_offset * 0x08080808) + (0x07060504 << 2);

  uint32_t exp_offsets[2] = {
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)};

  uint32_t res[2] = {
      lop3_and_or(qb << 4, 0x80808080, exp_offsets[0]),
      lop3_and_or(qb, 0x80808080, exp_offsets[1])};

  return *reinterpret_cast<uint2 *>(res);
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Int8>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t buffer1 = 0x03020100 << exp_offset;
  uint32_t buffer2 = 0x0C080604 << exp_offset;

  uint32_t res[2];
  uint32_t signs[2] = {qb >> 3, qb >> 7};
  uint32_t int8s[2] = {
      __byte_perm(buffer1, buffer2, qb),
      __byte_perm(buffer1, buffer2, qb >> 16)};

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = __byte_perm(int8s[0], int8s[1], 0x6420 + 0x1111 * i);
    uint32_t flag = signs[i] & 0x01010101;
    uint32_t mask = flag * 0xFF;
    res[i] = (val - flag) ^ mask;
  }

  return *reinterpret_cast<uint2 *>(res);
}

template <class TargetType, uint32_t kCount, bool kUseWgmma>
CUDA_INLINE void fused_dequant_for_mxfp4(const uint32_t *qb_ptrs, uint32_t *res_ptrs, uint32_t *scales_ptr) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t exp_offset = reinterpret_cast<uint8_t *>(scales_ptr)[i];
    uint2 res = fused_dequant_single_for_mxfp4<TargetType>(qb_ptrs[i], exp_offset);
    res_ptrs[i * 2] = res.x;
    res_ptrs[i * 2 + 1] = res.y;
  }

  if constexpr (kUseWgmma) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kCount; i++) {
      uint32_t tmp = res_ptrs[i * 4 + 1];
      res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
      res_ptrs[i * 4 + 2] = tmp;
    }
  }
}


template <uint32_t kFixedExpOffset>
CUDA_INLINE uint2 fixed_dequant_single_for_mxfp4_e4m3(const uint32_t qb) {
  static_assert(kFixedExpOffset == 1, "POC only validates fixed exp_offset=1");

  constexpr uint32_t exp_offset_buffer1 =
      (kFixedExpOffset * 0x08080800) + ((0x03020100 << 2) - 0x00000400);
  constexpr uint32_t exp_offset_buffer2 =
      (kFixedExpOffset * 0x08080808) + (0x07060504 << 2);

  uint32_t exp_offsets[2] = {
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)};

  uint32_t res[2] = {
      lop3_and_or(qb << 4, 0x80808080, exp_offsets[0]),
      lop3_and_or(qb, 0x80808080, exp_offsets[1])};

  return *reinterpret_cast<uint2 *>(res);
}


// Decode the native mode-2 MXFP4 payload with a compile-time exponent offset.
// Unlike fused_dequant_for_mxfp4, this decoder does not consume the resident
// raw relative scale. kFixedExpOffset=1 produces q / 32 in E4M3 registers.
template <class TargetType, uint32_t kCount, bool kUseWgmma, uint32_t kFixedExpOffset>
CUDA_INLINE void fixed_dequant_for_mxfp4(const uint32_t *qb_ptrs, uint32_t *res_ptrs) {
  static_assert(std::is_same<TargetType, Float8E4M3>::value);
  static_assert(kFixedExpOffset == 1, "POC only validates fixed exp_offset=1");

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint2 res = fixed_dequant_single_for_mxfp4_e4m3<kFixedExpOffset>(qb_ptrs[i]);
    res_ptrs[i * 2] = res.x;
    res_ptrs[i * 2 + 1] = res.y;
  }

  if constexpr (kUseWgmma) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kCount; i++) {
      uint32_t tmp = res_ptrs[i * 4 + 1];
      res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
      res_ptrs[i * 4 + 2] = tmp;
    }
  }
}
