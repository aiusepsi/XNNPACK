#include "jit/aarch32-assembler.h"

#include <string.h>

namespace aarch32 {
static const int DEFAULT_BUFFER_SIZE = 4096;

Assembler::Assembler() {
  buffer_ = new uint8_t[DEFAULT_BUFFER_SIZE];
  cursor_ = buffer_;
  top_ = buffer_ + DEFAULT_BUFFER_SIZE;
  error = kNoError;
}

Assembler::~Assembler() { delete[] buffer_; }

void Assembler::Emit32(uint32_t value) {
  if (error != kNoError) {
    return;
  }

  if (cursor_ + 4 > top_) {
    error = kOutOfSpace;
  }

  memcpy(cursor_, &value, sizeof(value));
  cursor_ += 4;
}

void Assembler::add(Register Rd, Register Rn, Register Rm) {
  Emit32(0x8 << 20 | Rn.code << 16 | Rd.code << 12 | Rm.code);
}

void Assembler::reset() { cursor_ = buffer_; }
}  // namespace aarch32
