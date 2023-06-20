#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ConcatOnnx);

ace::OpType ConcatOnnx::opType() { return ace::OpType_Concat; }
ace::OpParameter ConcatOnnx::type() { return ace::OpParameter_Axis; }

void ConcatOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) {
  auto para = new ace::AxisT;
  para->axis = 0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      para->axis = attributeProto.i();
    }
  }

  dstOp->main.value = para;
}

REGISTER_CONVERTER(ConcatOnnx, Concat);

}  // namespace converter
}  // namespace ace