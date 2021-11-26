// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/params.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>


class TransposeMicrokernelTester {
 public:
  inline TransposeMicrokernelTester& height(size_t height) {
    assert(height != 0);
    this->height_ = height;
    return *this;
  }

  inline size_t height() const { return this->height_; }

  inline TransposeMicrokernelTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const { return this->width_; }

  inline TransposeMicrokernelTester& offset(size_t offset) {
    assert(offset != 0);
    this->offset_ = offset;
    return *this;
  }

  inline size_t offset() const { return this->offset_; }

  inline TransposeMicrokernelTester& h_block_size(size_t h_block_size) {
    assert(h_block_size != 0);
    this->h_block_size_ = h_block_size;
    return *this;
  }

  inline size_t h_block_size() const { return this->h_block_size_; }

  inline TransposeMicrokernelTester& w_block_size(size_t w_block_size) {
    assert(w_block_size != 0);
    this->w_block_size_ = w_block_size;
    return *this;
  }

  inline size_t w_block_size() const { return this->w_block_size_; }

  inline TransposeMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const { return this->iterations_; }

  void Test(xnn_x32_transpose_ukernel_function transpose) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    std::vector<uint32_t> input(height() * width() + XNN_EXTRA_BYTES / sizeof(int32_t));
    std::vector<uint32_t> output(height() * width());
    size_t shape[] = {width(), height()};
    size_t perm[] = {1, 0};
    std::vector<uint32_t> output_ref(height() * width(), 0);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::fill(output.begin(), output.end(), 0);

      for (size_t i = 0; i < width() * height(); i++) input[i] = i;
      // Call optimized micro-kernel.
      transpose(input.data(), output.data(), offset(), height(), width(),
                h_block_size(), w_block_size(), nullptr /* params */);

      // Verify results.
      reference_transpose(input.data(), output_ref.data(), shape, perm,
                          2 /* num_dims */);
      for (size_t i = 0; i < output.size(); i++) {
        EXPECT_EQ(output_ref[i], output[i]);
      }
    }
  }

 private:
  void reference_transpose(uint32_t* input, uint32_t* output, size_t* shape, size_t* perm,
                           size_t num_dims) const {
    size_t size = 1;
    for (size_t i = 0; i < num_dims; ++i) {
      size *= shape[i];
    }
    std::vector<size_t> input_stride(num_dims, 1);
    std::vector<size_t> output_stride(num_dims, 1);
    for (size_t i = 1; i < num_dims; ++i) {
      input_stride[i] = input_stride[i - 1] * shape[i - 1];
      output_stride[perm[i]] = output_stride[perm[i - 1]] * shape[perm[i - 1]];
    }

    for (size_t i = 0; i < size; ++i) {
      size_t out_idx = 0;
      for (size_t j = 0; j < num_dims; ++j) {
        size_t in_idx = (i / input_stride[j]) % shape[j];
        out_idx += in_idx * output_stride[j];
      }
      output[out_idx] = input[i];
    }
  }
  size_t height_ = 1;
  size_t width_ = 1;
  size_t offset_ = 1;
  size_t h_block_size_ = 1;
  size_t w_block_size_ = 1;
  size_t iterations_ = 15;
};
