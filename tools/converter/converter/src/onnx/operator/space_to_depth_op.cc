#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(SpaceToDepthOnnx);

ace::OpType SpaceToDepthOnnx::opType() { return ace::OpType_SpaceToDepth; }

ace::OpParameter SpaceToDepthOnnx::type() {
  return ace::OpParameter_DepthSpaceParam;
}

void SpaceToDepthOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                           OnnxScope* scope) {
  auto spaceToDepthParam = new ace::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(SpaceToDepthOnnx, SpaceToDepth);

}  // namespace converter
}  // namespace ace