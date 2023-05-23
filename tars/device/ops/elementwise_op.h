#pragma once

#include "tars/device/kernels/x86/elementwise.h"
#include "tars/device/op_context.h"
#include "tars/device/operator.h"
#include "tars/ir/op_option_generated.h"
#include "tars/ir/types_generated.h"

namespace ace {
namespace device {

template <RuntimeType rtype, DataType dtype>
class ElementwiseOperator : public Operator<rtype, dtype> {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<dtype>::value_type T;
  // type alias
  using Inputs = std::vector<Tensor<dtype> *>;
  using Outputs = std::vector<Tensor<dtype> *>;

  ElementwiseOperator() {}

  virtual Status init(const OpContext &ctx, Inputs inputs,
                      Outputs outputs) override {
    this->name_ = "ElementwiseOperator";
    this->ctx_ = ctx;
    this->ctx_.dtype = dtype;
    this->ctx_.name = name_;

    this->inputs_ = inputs;
    this->outputs_ = outputs;
    return Status::OK();
  }

  virtual Status run() override {
    CHECK(rtype != RuntimeType::CUDA);

    auto inputs = this->inputs_[0]->mutable_data();
    auto outputs = this->outputs_[0]->mutable_data();
    auto size = this->inputs_[0]->shape().size();

    if (ctx_.name == "Abs") {
      x86::_avx2_abs(outputs, inputs, size);
    } else if (ctx_.name == "BoundedRelu") {
      auto option = this->ctx_.option.AsBoundedReluOption();
      x86::_avx2_bounded_relu(outputs, inputs, size, option->threshold);
    } else if (ctx_.name == "Clip") {
      auto option = this->ctx_.option.AsClipOption();
      x86::_avx2_clip(outputs, inputs, size, option->min, option->max);
    } else if (ctx_.name == "ClipV2") {
      auto option = this->ctx_.option.AsClipV2Option();
      x86::_avx2_clipv2(outputs, inputs, size, option->min, option->max);
    } else if (ctx_.name == "ClippedRelu") {
      auto option = this->ctx_.option.AsClippedReluOption();
      x86::_avx2_clipped_relu(outputs, inputs, size, option->min, option->max);
    } else if (ctx_.name == "Elu") {
      x86::_avx2_elu(outputs, inputs, size);
    } else if (ctx_.name == "Exp") {
      x86::_avx2_exp(outputs, inputs, size);
    } else if (ctx_.name == "GeluTanh") {
      x86::_avx2_gelu(outputs, inputs, size);
    } else if (ctx_.name == "HardSigmoid") {
      auto option = this->ctx_.option.AsHardSigmoidOption();
      x86::_avx2_hard_sigmoid(outputs, inputs, size, option->alpha,
                              option->beta);
    } else if (ctx_.name == "HardSwish") {
      auto option = this->ctx_.option.AsHardSwishOption();
      x86::_avx2_hard_swish(outputs, inputs, size, option->shift,
                            option->scale);
    } else if (ctx_.name == "LeakyRelu") {
      auto option = this->ctx_.option.AsLeakyReluOption();
      x86::_avx2_leaky_relu(outputs, inputs, size, option->alpha);
    } else if (ctx_.name == "Linear") {
      LOG(FATAL) << "Linear not implemented.";
    } else if (ctx_.name == "Log") {
      x86::_avx2_log(outputs, inputs, size);
    } else if (ctx_.name == "Logistic") {
      x86::_avx2_logistic(outputs, inputs, size);
    } else if (ctx_.name == "LogSigmoid") {
      x86::_avx2_log_sigmoid(outputs, inputs, size);
    } else if (ctx_.name == "Mish") {
      auto option = this->ctx_.option.AsMishOption();
      x86::_avx2_mish(outputs, inputs, size, option->scale);
    } else if (ctx_.name == "PRelu") {
      auto option = this->ctx_.option.AsPReluOption();
      x86::_avx2_prelu(outputs, inputs, size, option->slope);
    } else if (ctx_.name == "Relu") {
      x86::_avx2_relu(outputs, inputs, size);
    } else if (ctx_.name == "Relu6") {
      x86::_avx2_relu6(outputs, inputs, size);
    } else if (ctx_.name == "Round") {
      x86::_avx2_round(outputs, inputs, size);
    } else if (ctx_.name == "Selu") {
      x86::_avx2_selu(outputs, inputs, size);
    } else if (ctx_.name == "Sigmoid") {
      x86::_avx2_sigmoid(outputs, inputs, size);
    } else if (ctx_.name == "SoftRelu") {
      auto option = this->ctx_.option.AsSoftReluOption();
      x86::_avx2_soft_relu(outputs, inputs, size, option->threshold);
    } else if (ctx_.name == "SoftReluV2") {
      auto option = this->ctx_.option.AsSoftReluOption();
      x86::_avx2_soft_relu(outputs, inputs, size, option->threshold);
    } else if (ctx_.name == "Sqrt") {
      x86::_avx2_sqrt(outputs, inputs, size);
    } else if (ctx_.name == "Swish") {
      x86::_avx2_swish(outputs, inputs, size);
    } else if (ctx_.name == "Tanh") {
      auto option = this->ctx_.option.AsTanhOption();
      x86::_avx2_tanh(output, inputs, size, option->min, option->max);
    } else {
      LOG(FATAL) << "Fatal.";
    }

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
    virtual Status run() override {                                           \
      return ElementwiseOperator<rtype, data_type>::run();                    \
    }                                                                         \
    virtual Status shape_infer() override {                                   \
      return ElementwiseOperator<rtype, data_type>::shape_infer();            \
    }                                                                         \
  };  // namespace device

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