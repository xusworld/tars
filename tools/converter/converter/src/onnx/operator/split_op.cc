#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(SplitOnnx);

ace::OpType SplitOnnx::opType() { return ace::OpType_Slice; }

ace::OpParameter SplitOnnx::type() { return ace::OpParameter_Slice; }

void SplitOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
  auto param = new ace::SliceT;
  // Default axis = 0:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
  int axis = 0;
  std::vector<int> slicePoints;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      axis = attributeProto.i();
    } else if (attributeName == "split") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      const int splitSize = attributeProto.ints_size();
      for (int k = 0; k < splitSize; ++k) {
        slicePoints.push_back(attributeProto.ints(k));
      }
    }
  }
  param->axis = axis;
  param->slicePoints = slicePoints;
  param->sourceType = ace::NetSource_TENSORFLOW;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(SplitOnnx, Split);

}  // namespace converter
}  // namespace ace