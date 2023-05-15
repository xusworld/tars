#pragma once

#include "tars/core/status.h"
#include "tars/device/operator_impl.h"
#include "tars/ir/types_generated.h"

namespace ace {
namespace device {

template <RuntimeType rtype, DataType data_type>
class ElementwiseImpl : public OperatorImpl<rtype, data_type> {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  // type alias
  using Input = Tensor<data_type>;
  using Output = Tensor<data_type>;
  using Inputs = const std::vector<Input *>;
  using Outputs = std::vector<Output *>;

 public:
  virtual Status init(const OpParam &param, Inputs inputs,
                      Outputs outputs) override {
    return Status::OK();
  }

  virtual Status create(const OpParam &param, Inputs inputs,
                        Outputs outputs) override {
    LOG(INFO) << "ElementwiseImpl| create";
    return Status::OK();
  }

  virtual Status dispatch(const OpParam &param, Inputs inputs,
                          Outputs outputs) override {
    LOG(INFO) << "ElementwiseImpl| dispatch";
    std::string type = "Relu";
    if (type == "Relu") {
      auto option = param.option.AsReluOption();
      // single input, single output
      CHECK(inputs.size() == 1);
      auto in = inputs[0]->mutable_data();
      auto out = outputs[0]->mutable_data();

      LOG(INFO) << "inputs[0].size(): " << inputs[0]->size();
      for (int i = 0; i < inputs[0]->size(); i++) {
        out[i] = in[i] + 1;
      }
    }

    return Status::OK();
  }
};

}  // namespace device
}  // namespace ace