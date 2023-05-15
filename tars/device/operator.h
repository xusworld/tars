#pragma once

#include "tars/core/macro.h"
#include "tars/core/status.h"
#include "tars/core/tensor.h"
#include "tars/device/operator_impl.h"
#include "tars/device/operator_param.h"
#include "tars/ir/op_generated.h"
#include "tars/ir/op_option_generated.h"
#include "tars/ir/types_generated.h"

namespace ace {
namespace device {

template <RuntimeType rtype, DataType data_type>
class Operator {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  // type alias
  using Input = Tensor<data_type>;
  using Output = Tensor<data_type>;
  using Inputs = const std::vector<Input *>;
  using Outputs = std::vector<Output *>;

  Operator() = default;

  virtual ~Operator() = default;

  virtual Status init(const OpParam &op_param, Inputs inputs,
                      Outputs outputs) = 0;

  virtual Status invoke() = 0;

  virtual Status shape_infer() = 0;

  // returns the name of the operator
  std::string name() const { return ""; }

  // returns the desc of the operator
  std::string desc() const { return ""; }

  // returns the optype of the operator
  OpType op_type() const { return op_type_; }

  // return the datatype of the operator
  DataType dtype() const { return dtype_; }

  // returns the inputs number of the operator
  int32_t inputs_num() const { return inputs_.size(); }

  // returns the outputs number of the operator
  int32_t outputs_num() const { return outputs_.size(); }

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