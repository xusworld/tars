#pragma once

#include <cassert>

#include "jit_code_generator.h"

// Copy From weihuali
namespace kernels {
namespace cpu {
namespace exp {

namespace {

constexpr int kPack = 8;
constexpr int kUnroll = 8;

}  // namespace

class DotProductImpl : public JITCodeGenerator {
 public:
  DotProductImpl() = default;
  ~DotProductImpl() = default;

 public:
  void GenerateCode() override;
  float operator()(float *x, float *y, int n);

 private:
  int n_;
  const Xbyak::Reg64 &reg_x_ = rdi;
  const Xbyak::Reg64 &reg_y_ = rsi;
  const Xbyak::Reg64 &reg_n_ = rdx;
  const Xbyak::Reg64 &reg_output_ = rcx;
  const Xbyak::Reg64 &reg_m_ = r8;

  Xbyak::Reg64 reg_tmp_ = r9;
  Xbyak::Reg64 reg_loop_ = r10;

  Xbyak::Ymm px_ = Xbyak::Ymm(14);
  Xbyak::Ymm py_ = Xbyak::Ymm(15);
};

}  // namespace exp
}  // namespace cpu
}  // namespace kernels
