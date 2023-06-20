#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"
namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(FlattenOnnx);

ace::OpType FlattenOnnx::opType() { return ace::OpType_Flatten; }

ace::OpParameter FlattenOnnx::type() { return ace::OpParameter_Flatten; }

void FlattenOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope *scope) {
  auto param = new ace::FlattenT;

  // Ref https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten,
  // Default is 1
  int axis = 1;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "axis") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      axis = attributeProto.i();
    }
  }
  param->axis = axis;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(FlattenOnnx, Flatten);

}  // namespace converter
}  // namespace ace