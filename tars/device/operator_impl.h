#pragma once

#include "tars/core/status.h"
#include "tars/core/tensor.h"
#include "tars/core/types.h"
#include "tars/device/operator_param.h"
#include "tars/ir/types_generated.h"

namespace ace {
namespace device {

// operator impl 定义了
template <RuntimeType rtype, DataType data_type>
class OperatorImpl {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  // type alias
  using Input = Tensor<data_type>;
  using Output = Tensor<data_type>;
  using Inputs = const std::vector<Input *>;
  using Outputs = std::vector<Output *>;

  OperatorImpl() = default;
  virtual ~OperatorImpl() = default;

  virtual Status init(const OpParam &param, Inputs inputs, Outputs outputs) {
    // init operator
    LOG(INFO) << "OperatorImpl| init";
    return Status::OK();
  }

  virtual Status create(const OpParam &param, Inputs inputs,
                        Outputs outputs) = 0;

  virtual Status dispatch(const OpParam &param, Inputs inputs,
                          Outputs outputs) = 0;

 protected:
  std::string name_;
  Inputs inputs_;
  Outputs outputs_;
  OperatorImpl<rtype, data_type> *impl_;
};

}  // namespace device
}  // namespace ace