#pragma once

#include "tars/device/impl/x86/elementwise_impl.h"
#include "tars/device/operator.h"
#include "tars/device/operator_param.h"
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

template <RuntimeType rtype, DataType data_type>
class ElementwiseOperator final : public Operator<rtype, data_type> {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  // type alias
  using Input = Tensor<data_type>;
  using Output = Tensor<data_type>;
  using Inputs = std::vector<Input *>;
  using Outputs = std::vector<Output *>;

  ElementwiseOperator() {}

  virtual Status init(const OpParam &op_param, Inputs inputs,
                      Outputs outputs) override {
    LOG(INFO) << "ElementwiseOperator| init";
    this->name_ = "ElementwiseOperator";
    this->op_param_ = op_param;
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    this->impl_ = new ElementwiseImpl<rtype, data_type>;
    this->impl_->init(this->op_param_, this->inputs_, this->outputs_);
    return Status::OK();
  }

  virtual Status invoke() override {
    LOG(INFO) << "ElementwiseOperator| invoke";
    this->impl_->dispatch(this->op_param_, this->inputs_, this->outputs_);
    return Status::OK();
  }

  virtual Status shape_infer() override {
    LOG(INFO) << "ElementwiseOperator| shape inference";
    return Status::OK();
  }

  virtual ~ElementwiseOperator() { LOG(INFO) << "Do something here."; }

 protected:
  // basic op's attributes
  std::string name_;
  std::string desc_;
  OpType op_type_;
  DataType dtype_;
  OpParam op_param_;
  OpOption op_option_;

  Inputs inputs_;
  Outputs outputs_;
  OperatorImpl<rtype, data_type> *impl_;
};

}  // namespace device
}  // namespace ace