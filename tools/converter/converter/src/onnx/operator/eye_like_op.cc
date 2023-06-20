#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(EyeLikeOnnx);

ace::OpType EyeLikeOnnx::opType() { return ace::OpType_EyeLike; }
ace::OpParameter EyeLikeOnnx::type() { return ace::OpParameter_NONE; }

void EyeLikeOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope *scope) {
  int k = 0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attr = onnxNode->attribute(i);
    const auto &attrName = attr.name();
    if (attrName == "k") {
      k = attr.i();
    }
  }
  dstOp->inputIndexes.push_back(
      scope->buildIntConstOp({k}, dstOp->name + "/k"));
}

REGISTER_CONVERTER(EyeLikeOnnx, EyeLike);

}  // namespace converter
}  // namespace ace