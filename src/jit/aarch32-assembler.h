#include <cstdint>

namespace aarch32 {
struct Register {
  uint8_t code;
};

const Register r0{0};
const Register r1{1};
const Register r2{2};
const Register r3{3};
const Register r4{4};
const Register r5{5};
const Register r6{6};
const Register r7{7};
const Register r8{8};
const Register r9{9};
const Register r10{10};
const Register r11{11};
const Register r12{12};
const Register r13{13};
const Register r14{14};
const Register r15{15};
const Register sp = r13;
const Register lr = r14;
const Register pc = r15;

enum Error {
  kNoError,
  kOutOfSpace,
};

// A simple AAarch32 assembler.
// Right now it allocates its own memory (using `new`) to write code into (for
// testing), but will be updated to be more customizable.
class Assembler {
 public:
  explicit Assembler();
  ~Assembler();

  void add(Register Rd, Register Rn, Register Rm);
  void reset();

  const uint8_t* const start() { return buffer_; }

  void Emit32(uint32_t value);

  // Pointer to start of memory region to write code.
  uint8_t* buffer_;
  // Pointer to current place in memory region.
  uint8_t* cursor_;
  // Pointer to out-of-bounds of memory region.
  uint8_t* top_;
  // Errors encountered while assembling code.
  Error error;
};
}  // namespace aarch32
