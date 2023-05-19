#include <xbyak/xbyak.h>

#include <vector>

#include "abi.h"

namespace kernels {
namespace cpu {
namespace exp {

constexpr int kMaxCodeSize = 256 * 1024;
constexpr const int kRegisterWidthInBytes = sizeof(void *);

class JITCodeGenerator : public Xbyak::CodeGenerator {
 public:
  JITCodeGenerator(size_t code_size = kMaxCodeSize)
      : Xbyak::CodeGenerator(code_size, Xbyak::AutoGrow){};

  virtual ~JITCodeGenerator() = default;

 public:
  // Create a kernel
  virtual void CreateKernel() {
    GenerateCode();
    jit_kernel_ = GetCode();
    assert(jit_kernel_ != nullptr);
  }

  // Function Call
  template <typename... KernelArgs>
  void operator()(KernelArgs... args) const {
    using Func = void (*)(const KernelArgs... args);
    auto *fptr = (Func)jit_kernel_;
    (*fptr)(std::forward<KernelArgs>(args)...);
  }

  void Prolog() {
    // 1. save base stack pointer
    push(abi::RBP);
    mov(abi::RBP, abi::RSP);

    /*
        push(a) does the following things:
            rsp -= sizeof(void *)
            rsp = (a)
        so, after mov(xbp, xsp), [xbp] is the xbp of caller frame
        next pushed object will be at -4[xbp] or -8[xbp]
    */
    abi_bp_offset_ = -abi::RegisterWidthInBytes;
    // 2. save the regs that abi require callee to save, only windows x64
    for (int i = 0; i < abi::ABICalleeSaveRegsNum; i++) {
      push(Xbyak::Reg64(abi::ABICalleeSaveRegs[i]));
      abi_bp_offset_ -= abi::RegisterWidthInBytes;
    }

    // 3. save the arguements from register to stack
    size_t abi_stack_arg_offset =
        abi::ABIStackParamOffset + 2 * abi::RegisterWidthInBytes;

    for (int i = 0; i < abi_nb_argment; i++) {
      if (i < abi::ABIArgsRegsNum) {
        push(Xbyak::Reg64(abi::ABIArgsRegs[i]));
        arguement_offsets_.push_back(abi_bp_offset_);
        abi_bp_offset_ -= abi::RegisterWidthInBytes;
      } else {
        arguement_offsets_.push_back(abi::ABIStackParamOffset);
        abi_stack_arg_offset += abi::RegisterWidthInBytes;
      }
    }
  }

  void Epilog() {
    // 1. rewind the stack for function arguments
    size_t size_in_bytes = 0;
    for (int i = 0; i < abi_nb_argment; i++) {
      if (i < abi::ABIArgsRegsNum) {
        size_in_bytes += abi::RegisterWidthInBytes;
      }
    }

    if (size_in_bytes > 0) add(abi::RSP, size_in_bytes);

    // 2. restore the regs that abi require callee to save
    for (int i = abi::ABICalleeSaveRegsNum - 1; i >= 0; i--) {
      pop(Xbyak::Reg64(abi::ABICalleeSaveRegs[i]));
    }
    // 3. restore the base stack pointer
    leave();  // mov(sp, bp); pop(bp);
  }

  void uni_vzeroupper() {
    // if (cpu_has_isa(AVX)) vzeroupper();
    vzeroupper();
  }

 protected:
  virtual void GenerateCode() = 0;
  const Xbyak::uint8 *jit_kernel_;

 private:
  size_t abi_nb_argment = 0;
  size_t abi_bp_offset_ = 0;
  size_t stack_variable_offset_ = 0;

  std::vector<size_t> arguement_offsets_;

  const Xbyak::uint8 *GetCode() {
    // Xbyak::AutoGrow should call ready() before to run!
    // mode = Read/Write/Exec
    this->ready();
    const Xbyak::uint8 *code = CodeGenerator::getCode();
    return code;
  }
};

}  // namespace exp
}  // namespace cpu
}  // namespace kernels