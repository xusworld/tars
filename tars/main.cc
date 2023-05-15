#include <iostream>

#include "glog/logging.h"
#include "tars/core/macro.h"
#include "tars/core/tensor.h"
#include "tars/core/tensor_shape.h"
#include "tars/core/types.h"
#include "tars/device/ops/elementwise_op.h"
#include "tars/ir/types_generated.h"

int main() {
  // Elementwise Op test
  LOG(INFO) << "Elementwise Op test";
  // 创建待测试的 tensor shape
  ace::TensorShape shape = ace::TensorShape({32, 224, 224, 3});

  // 创建保存输入数据的 tensors
  ace::Tensor<int32> in(ace::RuntimeType::CPU, shape);
  LOG(INFO) << "Input tensor shape info: " << in.shape();
  in.reset(0);
  auto in_data = in.data();
  LOG(INFO) << "input data: " << in_data[0];

  ace::Tensor<int32> out(ace::RuntimeType::CPU);

  ace::device::OpParam op_param;

  auto inputs = std::vector<ace::Tensor<int32>*>({&in});
  auto outputs = std::vector<ace::Tensor<int32>*>({&out});

  ace::device::ElementwiseOperator<ace::RuntimeType::CPU, int32> op;
  op.init(op_param, inputs, outputs);
  op.invoke();

  auto data = out.data();
  LOG(INFO) << "Output: " << data[0];
  return 0;
}