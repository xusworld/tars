#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(TransposeOnnx);

ace::OpType TransposeOnnx::opType() { return ace::OpType_Permute; }

ace::OpParameter TransposeOnnx::type() { return ace::OpParameter_Permute; }

void TransposeOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                        OnnxScope *scope) {
  auto param = new ace::PermuteT;

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "perm") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      param->dims.resize(attributeProto.ints_size());
      for (int v = 0; v < attributeProto.ints_size(); ++v) {
        param->dims[v] = attributeProto.ints(v);
      }
    }
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(TransposeOnnx, Transpose);

}  // namespace converter
}  // namespace ace