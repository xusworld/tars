#include <stdio.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(SqueezeOnnx);

ace::OpType SqueezeOnnx::opType() { return ace::OpType_Squeeze; }
ace::OpParameter SqueezeOnnx::type() { return ace::OpParameter_SqueezeParam; }

void SqueezeOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
  auto para = new ace::SqueezeParamT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axes") {
      para->squeezeDims.resize(attributeProto.ints_size());
      for (int i = 0; i < para->squeezeDims.size(); ++i) {
        para->squeezeDims[i] = attributeProto.ints(i);
      }
    }
  }

  dstOp->main.value = para;
}

REGISTER_CONVERTER(SqueezeOnnx, Squeeze);

}  // namespace converter
}  // namespace ace