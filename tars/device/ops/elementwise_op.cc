#include "tars/device/kernels/x86/elementwise.h"
#include "tars/device/ops/elementwise_op.h"

namespace ace {
namespace device {

std::string ElementwiseOpTypeToString(OpType type) {
  switch (type) {
    case OpType_Abs:
      return "Abs";
    case OpType_BoundedRelu:
      return "BoundedRelu";
    case OpType_Clip:
      return "Clip";
    case OpType_ClipV2:
      return "ClipV2";
    case OpType_ClippedRelu:
      return "ClippedRelu";
    case OpType_Elu:
      return "Elu";
    case OpType_Exp:
      return "Exp";
    case OpType_GeluTanh:
      return "GeluTanh";
    case OpType_HardSigmoid:
      return "HardSigmoid";
    case OpType_HardSwish:
      return "HardSwish";
    case OpType_LeakyRelu:
      return "LeakyRelu";
    case OpType_Linear:
      return "Linear";
    case OpType_Log:
      return "Log";
    case OpType_Logistic:
      return "Logistic";
    case OpType_Mish:
      return "Mish";
    case OpType_Pow:
      return "Pow";
    case OpType_PRelu:
      return "PRelu";
    case OpType_Relu:
      return "Relu";
    case OpType_Relu6:
      return "Relu6";
    case OpType_Round:
      return "Round";
    case OpType_Selu:
      return "Selu";
    case OpType_Sigmoid:
      return "Sigmoid";
    case OpType_SoftRelu:
      return "SoftRelu";
    case OpType_SoftReluV2:
      return "SoftReluV2";
    case OpType_Sqrt:
      return "Sqrt";
    case OpType_Swish:
      return "Swish";
    case OpType_Tanh:
      return "Tanh";
    default:
      LOG(INFO) << "Got a error type, please check.";
      return "";
  }
}

template <typename Func, typename Vec, int pack>
void SimdElementwiseExec(std::vector<float> &args, void *outputs_ptr,
                         void *inputs_ptr, const int size) {
  Func func;
  int quotient = size / pack;
  int remainder = size - quotient * pack;

  auto inputs = (const float *)inputs_ptr;
  auto outputs = (float *)outputs_ptr;

#pragma omp parallel for
  for (int i = 0; i < size; i += pack) {
    Vec src;
    src.load(inputs + i);

    Vec dst(func(src, args));
    dst.store(outputs + i);
  }

  // if (remainder > 0) {
  //   float left_inputs[pack];
  //   float left_outputs[pack];
  //   memcpy(left_inputs, inputs, remainder * sizeof(float));
  //   Vec left_src;
  //   left_src.load(left_inputs);
  //   Vec left_dst(func(left_src));
  //   left_dst.store(left_outputs);
  //   memcpy(outputs, left_outputs, remainder * sizeof(float));
  // }
}

// template void SimdElementwiseExec<x86::ElementwiseAbs<vector::Vec8f>,
//                                   vector::Vec8f, 8>(std::vector<float> &args,
//                                                     void *outputs_ptr,
//                                                     void *inputs_ptr,
//                                                     const int size);

}  // namespace device
}  // namespace ace
