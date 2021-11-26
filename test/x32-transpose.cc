// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_4X4, block_4_4) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(4)
      .width(4)
      .h_block_size(4)
      .w_block_size(4)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_32X32, block_32_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(32)
      .width(32)
      .h_block_size(32)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_64X32, block_64_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(64)
      .width(32)
      .h_block_size(64)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_18X32, block_18_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(18)
      .width(32)
      .h_block_size(18)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_32X18, block_32_18) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(32)
      .width(18)
      .h_block_size(32)
      .w_block_size(18)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE_19X30, block_19_30) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(19)
      .width(30)
      .h_block_size(19)
      .w_block_size(30)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
