#pragma once

#include "tars/device/op_context.h"
#include "tars/device/operator.h"
#include "tars/ir/op_option_generated.h"
#include "tars/ir/types_generated.h"

namespace tars {
namespace device {

template <RuntimeType rtype, DataType dtype>
class ConcatOp : public Operator<rtype, dtype> {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<dtype>::value_type T;
  // type alias
  using Inputs = std::vector<Tensor<dtype> *>;
  using Outputs = std::vector<Tensor<dtype> *>;

  ConcatOp() = default;

  virtual ~ConcatOp() = default;

  virtual Status init(const OpContext &ctx, Inputs inputs,
                      Outputs outputs) override {
    this->name_ = "Reshape";
    this->desc_ = "a reshape operator";

    this->ctx_ = ctx;
    this->ctx_.dtype = dtype;
    this->ctx_.name = name_;

    this->inputs_ = inputs;
    this->outputs_ = outputs;
    return Status::OK();
  }

  virtual Status run(const TensorShape &shape) {
    // ConcatOp with a single input and a single output.
    CHECK(inputs_.size() == 1);
    CHECK(outputs_.size() == 1);

    this->shape_ = shape;

    // ConcatOp's input and output should share the same buffer.
    return Status::OK();
  }

  virtual Status shape_infer() override {
    TensorShape outshape;

    return Status::OK();
  }

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

}  // namespace device
}  // namespace tars