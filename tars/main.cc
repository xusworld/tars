#include <iostream>

#include "glog/logging.h"
#include "tars/core/macro.h"
#include "tars/core/tensor.h"
#include "tars/core/tensor_shape.h"
#include "tars/core/types.h"
#include "tars/device/ops/elementwise_op.h"
#include "tars/ir/types_generated.h"

// void Relu(float lhs) { std::cout << "Relu" << std::endl; }

// void Selu(float lhs, float rhs) { std::cout << "Selu" << std::endl; }

// template <typename Vec, typename... Args>
// void func(Vec vec, Args... args) {
//   using Func = void (*)(const Args... args);
//   auto* fptr = (Func)Relu;
//   (*fptr)(std::forward<Args>(args)...);
// }

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

  ace::device::OpContext op_param;

  auto inputs = std::vector<ace::Tensor<int32>*>({&in});
  auto outputs = std::vector<ace::Tensor<int32>*>({&out});

  ace::device::ReluOperator<ace::RuntimeType::CPU, int32> relu;
  relu.init(op_param, inputs, outputs);
  relu.run();

  auto data = out.data();
  LOG(INFO) << "Output: " << data[0];
  return 0;
}