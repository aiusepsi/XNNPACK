#include <jit/aarch32-assembler.h>

#include <gtest/gtest.h>

#define EXPECT_INSTR_EQ(expected, call) \
  a.reset();                            \
  call;                                 \
  memcpy(&actual, a.buffer_, 4);        \
  EXPECT_EQ(expected, actual)

namespace aarch32 {
TEST(AArch32Assembler, DataProcessingRegister) {
  Assembler a;
  uint32_t actual;
  EXPECT_INSTR_EQ(0x00810002, a.add(r0, r1, r2));
}
}  // namespace aarch32
