#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ReluOnnx);

ace::OpType ReluOnnx::opType() { return ace::OpType_ReLU; }
ace::OpParameter ReluOnnx::type() { return ace::OpParameter_Relu; }

void ReluOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  auto relu = new ace::ReluT;

  float slope = 0.01f;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();

    if (attributeName == "alpha") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      slope = attributeProto.f();
    } else {
      DLOG(ERROR) << "TODO!";
    }
  }

  if (onnxNode->op_type() == "LeakyRelu") {
    relu->slope = slope;
  } else {
    relu->slope = .0f;
  }

  dstOp->main.value = relu;
}

REGISTER_CONVERTER(ReluOnnx, Relu);
REGISTER_CONVERTER(ReluOnnx, LeakyRelu);

}  // namespace converter
}  // namespace ace