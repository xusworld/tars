#pragma once

#include <immintrin.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <xbyak/xbyak.h>
#include <xmmintrin.h>

#include <fstream>

namespace kernels {
namespace cpu {
namespace jit {
namespace abi {
#ifdef XBYAK64

constexpr Xbyak::Operand::Code abi_regs_callee_save[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::R12, Xbyak::Operand::R13,
    Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif  // _WIN32
};

constexpr Xbyak::Operand::Code abi_args_in_register[] = {
#ifdef _WIN32
    Xbyak::Operand::RCX,
    Xbyak::Operand::RDX,
    Xbyak::Operand::R8,
    Xbyak::Operand::R9,
#else
    Xbyak::Operand::RDI, Xbyak::Operand::RSI, Xbyak::Operand::RDX,
    Xbyak::Operand::RCX, Xbyak::Operand::R8,  Xbyak::Operand::R9,
#endif
};

constexpr const int abi_nb_args_in_register =
    sizeof(abi_args_in_register) / sizeof(Xbyak::Operand::Code);

#ifdef _WIN32
constexpr const int abi_stack_param_offset = 32;
#else
constexpr const int abi_stack_param_offset = 0;
#endif

#else   // XBYAK64

// ------------------------------------  32 bit related abi info
// ----------------------------------------

constexpr Xbyak::Operand::Code abi_regs_callee_save[] = {
    Xbyak::Operand::EBX,
    Xbyak::Operand::EDI,
    Xbyak::Operand::ESI,
};

static Xbyak::Operand::Code* abi_args_in_register;

constexpr const int abi_nb_args_in_register = 0;
constexpr const int abi_stack_param_offset = 0;
#endif  // XBYAK64

constexpr const int abi_nb_regs_callee_save =
    sizeof(abi_regs_callee_save) / sizeof(Xbyak::Operand::Code);
constexpr const int register_width_in_bytes = sizeof(void*);
// Change by luke
// constexpr const int register_width_in_bytes = sizeof(float*);

}  // namespace abi
}  // namespace jit
}  // namespace cpu
}  // namespace kernels
