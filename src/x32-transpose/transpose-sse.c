// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <emmintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__4x4_sse(
    const uint32_t *input,
    uint32_t * output,
    size_t offset,
    size_t h,
    size_t w,
    size_t h_block_size,
    size_t w_block_size,
    const void *params){
  const size_t ukernel_size = 4;
  assert(h_block_size >= ukernel_size);
  assert(w_block_size >= ukernel_size);
  assert(h >= h_block_size);
  assert(w >= w_block_size);
  size_t h_bytes = h * sizeof(uint32_t);
  size_t w_bytes = w * sizeof(uint32_t);
  size_t w_size = w_block_size;
  for (; w_size >= ukernel_size; w_size -= ukernel_size) {
    const int32_t* in_ptr_j = input;
    int32_t* out_ptr_j = output;
    size_t h_size = h_block_size;
    for (; h_size >= ukernel_size; h_size -= ukernel_size) {
      // Registers must be cast to __m128 as _MM_TRANSPOSE4_PS only accepts
      // __m128. This generates _mm_movelh_ps and _mm_movehl_ps which are not
      // available for __m128i.
      __m128 v0 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v1 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v2 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v3 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      int32_t *out_ptr = out_ptr_j;
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v0));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v1));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v2));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v3));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + ukernel_size * sizeof(uint32_t));
    }
    if XNN_UNLIKELY(h_size != 0) {
      // Shift input and output pointers back.
      const size_t address_increment = (ukernel_size - h_size) * sizeof(uint32_t);
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j - w * address_increment);
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j - address_increment);
      __m128 v0 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v1 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v2 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v3 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v0));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v1));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v2));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v3));
    }
    input = (int32_t*) ((uintptr_t) input + offset * ukernel_size * sizeof(uint32_t));
    output = (int32_t*) ((uintptr_t) output + h * ukernel_size * sizeof(uint32_t));
  }
  if XNN_UNLIKELY(w_size != 0) {
    // Shift input and output pointers back.
    const size_t h_address_increment = (ukernel_size - w_size) * sizeof(uint32_t);
    const int32_t* in_ptr_j = (int32_t*) ((uintptr_t) input -  h_address_increment);
    int32_t* out_ptr_j = (int32_t*) ((uintptr_t) output -  h_address_increment * h);
    size_t h_size = h_block_size;
    for (; h_size >= ukernel_size; h_size -= ukernel_size) {
      __m128 v0 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v1 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v2 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v3 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      int32_t *out_ptr = out_ptr_j;
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v0));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v1));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v2));
      out_ptr = (int32_t*) ((uintptr_t) out_ptr + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr), _mm_castps_si128(v3));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + ukernel_size * sizeof(uint32_t));
    }
    if XNN_UNLIKELY(h_size != 0) {
      // Shift input and output pointers back.
      const size_t w_address_increment = (ukernel_size - h_size) * sizeof(uint32_t);
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j - w_address_increment * w);
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j - w_address_increment);
      __m128 v0 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v1 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v2 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      in_ptr_j = (int32_t*) ((uintptr_t) in_ptr_j + w_bytes);
      __m128 v3 = _mm_castsi128_ps(_mm_loadu_si128((const __m128i*)((uintptr_t)in_ptr_j)));
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v0));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v1));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v2));
      out_ptr_j = (int32_t*) ((uintptr_t) out_ptr_j + h_bytes);
      _mm_storeu_si128((__m128i*)((uintptr_t)out_ptr_j), _mm_castps_si128(v3));
    }
  }
}
