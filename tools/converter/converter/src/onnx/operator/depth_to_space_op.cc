#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(DepthToSpaceOnnx);

ace::OpType DepthToSpaceOnnx::opType() { return ace::OpType_DepthToSpace; }

ace::OpParameter DepthToSpaceOnnx::type() {
  return ace::OpParameter_DepthSpaceParam;
}

void DepthToSpaceOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                           OnnxScope* scope) {
  auto spaceToDepthParam = new ace::DepthSpaceParamT;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "blocksize") {
      spaceToDepthParam->blockSize = (int)attributeProto.i();
    } else if (attributeName == "mode") {
      std::map<const std::string, ace::DepthToSpaceMode> strToMode = {
          {"DCR", ace::DepthToSpaceMode_DCR},
          {"CRD", ace::DepthToSpaceMode_CRD}};
      const std::string& modeStr = attributeProto.s();
      if (strToMode.find(modeStr) != strToMode.end()) {
        spaceToDepthParam->mode = strToMode[modeStr];
      } else {
        // ace_ERROR("ONNX DepthToSpace mode [%s] is currently not
        // supported.\n",
        //           modeStr.c_str());
      }
    }
  }

  dstOp->main.value = spaceToDepthParam;
}

REGISTER_CONVERTER(DepthToSpaceOnnx, DepthToSpace);

}  // namespace converter
}  // namespace ace