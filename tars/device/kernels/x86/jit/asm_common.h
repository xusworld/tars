#pragma once

#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <xbyak/xbyak.h>
#include <xmmintrin.h>

#include <fstream>
#include <vector>

namespace kernels {
namespace cpu {
namespace jit {
namespace common {

#ifdef XBYAK64
// ------------------------------------  64 bit related
// ----------------------------------------

#define rword_ Xbyak::util::qword

static Xbyak::Reg64 bp(Xbyak::Operand::RBP);
static Xbyak::Reg64 sp(Xbyak::Operand::RSP);

const std::vector<Xbyak::Reg64> regs_ = {
    Xbyak::util::rax,  // 1st return register, number of vector registers used
    Xbyak::util::rbx,  // callee-saved register; base poninters
    Xbyak::util::rcx,  // used to pass 4th integer argument to functions
    Xbyak::util::rdx,  // used to pass 3rd argument to functions; 2nd return
                       // register
    // Xbyak::util::rsp, // stack pointer
    // Xbyak::util::rbp, // callee-saved register, frame pointer
    Xbyak::util::rsi,  // used to pass 2nd argument to functions
    Xbyak::util::rdi,  // used to pass 1st argument to functions
    Xbyak::util::r8,   // used to pass 5th argument to functions
    Xbyak::util::r9,   // used to pass 6th argument to functions
    Xbyak::util::r10,  // temp register, used for passing a function's static
                       // chain ptr
    Xbyak::util::r11,  // temp register
    Xbyak::util::r12,  // callee-saved register
    Xbyak::util::r13,  // callee-saved register
    Xbyak::util::r14,  // callee-saved register
    Xbyak::util::r15,  // callee-saved register
};

#else  // XBYAK64
// ------------------------------------  32 bit related
// ----------------------------------------

typedef Xbyak::Reg32 xybak::Reg64;
#define rword_ Xbyak::util::dword

static xybak::Reg64 bp(Xbyak::Operand::EBP);
static xybak::Reg64 sp(Xbyak::Operand::ESP);

const std::vector<Xbyak::Reg32> regs_ = {
    Xbyak::util::eax,
    Xbyak::util::ebx,
    Xbyak::util::ecx,
    Xbyak::util::edx,
    /*Xbyak::util::esp, Xbyak::util::ebp,*/ Xbyak::util::esi,
    Xbyak::util::edi,
};

#endif  // XBYAK64

const std::vector<Xbyak::Mmx> mmx_ = {
    Xbyak::Mmx(0),  Xbyak::Mmx(1),  Xbyak::Mmx(2),
    Xbyak::Mmx(3),  Xbyak::Mmx(4),  Xbyak::Mmx(5),
    Xbyak::Mmx(6),  Xbyak::Mmx(7),
#ifdef XBYAK64
    Xbyak::Mmx(8),  Xbyak::Mmx(9),  Xbyak::Mmx(10),
    Xbyak::Mmx(11), Xbyak::Mmx(12), Xbyak::Mmx(13),
    Xbyak::Mmx(14), Xbyak::Mmx(15)
#endif
};

// const std::vector<Xbyak::Xmm> xmm_ = {
//     Xbyak::Xmm(0), Xbyak::Xmm(1), Xbyak::Xmm(2), Xbyak::Xmm(3),
//     Xbyak::Xmm(4), Xbyak::Xmm(5), Xbyak::Xmm(6), Xbyak::Xmm(7),
// #ifdef XBYAK64
//     Xbyak::Xmm(8), Xbyak::Xmm(9), Xbyak::Xmm(10),Xbyak::Xmm(11),
//     Xbyak::Xmm(12),Xbyak::Xmm(13),Xbyak::Xmm(14),Xbyak::Xmm(15)
// #endif
// };

// const std::vector<Xbyak::Ymm> ymm_ = {
//     Xbyak::Ymm(0), Xbyak::Ymm(1), Xbyak::Ymm(2), Xbyak::Ymm(3),
//     Xbyak::Ymm(4), Xbyak::Ymm(5), Xbyak::Ymm(6), Xbyak::Ymm(7),
// #ifdef XBYAK64
//     Xbyak::Ymm(8), Xbyak::Ymm(9), Xbyak::Ymm(10),Xbyak::Ymm(11),
//     Xbyak::Ymm(12),Xbyak::Ymm(13),Xbyak::Ymm(14),Xbyak::Ymm(15)
// #endif
// };

}  // namespace common
}  // namespace jit
}  // namespace cpu
}  // namespace kernels
