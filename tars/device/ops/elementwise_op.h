#pragma once

#include "tars/device/kernels/x86/elementwise.h"
#include "tars/device/op_context.h"
#include "tars/device/operator.h"
#include "tars/ir/op_option_generated.h"
#include "tars/ir/types_generated.h"

namespace ace {
namespace device {

/********************
Elementwise operatos:
  Abs,
  BoundedRelu,
  Clip,
  ClipV2,
  ClippedRelu,
  Elu,
  Exp,
  GeluTanh,
  HardSigmoid,
  HardSwish,
  LeakyRelu,
  Linear,
  Log,
  Logistic,
  LogSigmoid,
  Mish,
  Pow,
  PRelu,
  Relu,
  Relu6,
  Round,
  Selu,
  Sigmoid,
  SoftRelu,
  SoftReluV2,
  Sqrt,
  Swish,
  Tanh,
********************/

// template <typename Vec, typename... Args>
// using Func = void (*)(Vec vec, const Args... args);

// template <typename Vec, typename... Args>
// Func<Vec, Args...> ElementwiseOpTypeToFunc(OpType type) {
//   switch (type) {
//     case OpType_Abs:
//       return x86::abs;
//     case OpType_Relu:
//       return x86::relu;
//   }
// }

template <typename Func, typename Vec, int pack>
void SimdElementwiseExec(std::vector<float> &args, void *outputs_ptr,
                         void *inputs_ptr, const int size);

template <RuntimeType rtype, DataType data_type>
class ElementwiseOperator : public Operator<rtype, data_type> {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  // type alias
  using Inputs = std::vector<Tensor<data_type> *>;
  using Outputs = std::vector<Tensor<data_type> *>;

  ElementwiseOperator() {}

  virtual Status init(const OpContext &ctx, Inputs inputs,
                      Outputs outputs) override {
    this->name_ = "ElementwiseOperator";
    this->ctx_ = ctx;
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    return Status::OK();
  }

  virtual Status invoke() override {
    LOG(INFO) << "name_ : " << name_;
    this->op_param_.dtype = dtype_;
    this->op_param_.name = name_;

    auto in = this->inputs_[0]->mutable_data();
    auto out = this->outputs_[0]->mutable_data();

    if (ctx_.name == "Abs") {
      std::vector<float> args;
      // 使用类型萃取优化模型参数
      // SimdElementwiseExec<x86::ElementwiseAbs<vector::Vec8f>, vector::Vec8f,
      // 8>( args, out, in, 100);
    }

    // auto *func = ElementwiseOpTypeToFunc<std::vector<int>, float>();
    // (*fptr)(std::forward<Args>(args)...);
    return Status::OK();
  }

  virtual Status shape_infer() override {
    this->outputs_[0]->reshape(this->inputs_[0]->shape());
    return Status::OK();
  }

  virtual ~ElementwiseOperator() = default;

 protected:
  // basic op's attributes
  std::string name_;
  std::string desc_;
  OpType op_type_;
  DataType dtype_;
  OpContext ctx_;
  OpOption op_option_;

  Inputs inputs_;
  Outputs outputs_;
};

std::string ElementwiseOpTypeToString(OpType type);

#define DEFINE_ELEMENTWISE_OP(name, type)                                     \
  template <RuntimeType rtype, DataType data_type>                            \
  class name##Operator final : public ElementwiseOperator<rtype, data_type> { \
   public:                                                                    \
    using Inputs = std::vector<Tensor<data_type> *>;                          \
    using Outputs = std::vector<Tensor<data_type> *>;                         \
    virtual Status init(const OpContext &op_param, Inputs inputs,             \
                        Outputs outputs) {                                    \
      ElementwiseOperator<rtype, data_type>::init(op_param, inputs, outputs); \
      this->name_ = ElementwiseOpTypeToString(type);                          \
      return Status::OK();                                                    \
    }                                                                         \
  };

DEFINE_ELEMENTWISE_OP(Abs, OpType_Abs);
DEFINE_ELEMENTWISE_OP(BoundedRelu, OpType_BoundedRelu);
DEFINE_ELEMENTWISE_OP(Clip, OpType_Clip);
DEFINE_ELEMENTWISE_OP(ClipV2, OpType_ClipV2);
DEFINE_ELEMENTWISE_OP(ClippedRelu, OpType_ClippedRelu);
DEFINE_ELEMENTWISE_OP(Elu, OpType_Elu);
DEFINE_ELEMENTWISE_OP(Exp, OpType_Exp);
DEFINE_ELEMENTWISE_OP(GeluTanh, OpType_GeluTanh);
DEFINE_ELEMENTWISE_OP(HardSigmoid, OpType_HardSigmoid);
DEFINE_ELEMENTWISE_OP(HardSwish, OpType_HardSwish);
DEFINE_ELEMENTWISE_OP(LeakyRelu, OpType_LeakyRelu);
DEFINE_ELEMENTWISE_OP(Linear, OpType_Linear);
DEFINE_ELEMENTWISE_OP(Log, OpType_Log);
DEFINE_ELEMENTWISE_OP(Logistic, OpType_Logistic);
DEFINE_ELEMENTWISE_OP(LogSigmoid, OpType_LogSigmoid);
DEFINE_ELEMENTWISE_OP(Mish, OpType_Mish);
DEFINE_ELEMENTWISE_OP(Pow, OpType_Pow);
DEFINE_ELEMENTWISE_OP(PRelu, OpType_PRelu);
DEFINE_ELEMENTWISE_OP(Relu, OpType_Relu);
DEFINE_ELEMENTWISE_OP(Relu6, OpType_Relu6);
DEFINE_ELEMENTWISE_OP(Round, OpType_Round);
DEFINE_ELEMENTWISE_OP(Selu, OpType_Selu);
DEFINE_ELEMENTWISE_OP(Sigmoid, OpType_Sigmoid);
DEFINE_ELEMENTWISE_OP(SoftRelu, OpType_SoftRelu);
DEFINE_ELEMENTWISE_OP(SoftReluV2, OpType_SoftReluV2);
DEFINE_ELEMENTWISE_OP(Sqrt, OpType_Sqrt);
DEFINE_ELEMENTWISE_OP(Swish, OpType_Swish);
DEFINE_ELEMENTWISE_OP(Tanh, OpType_Tanh);

}  // namespace device
}  // namespace ace