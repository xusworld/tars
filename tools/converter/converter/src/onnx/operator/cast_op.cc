#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {
DECLARE_OP_CONVERTER(CastOnnx);

ace::OpType CastOnnx::opType() { return ace::OpType_Cast; }
ace::OpParameter CastOnnx::type() { return ace::OpParameter_CastParam; }

void CastOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  std::unique_ptr<ace::CastParamT> castParam(new ace::CastParamT);

  // not to use srcT parameter!
  castParam->srcT = ace::DataType_MAX;

  ::onnx::TensorProto_DataType castTo = ::onnx::TensorProto_DataType_UNDEFINED;
  const int attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "to") {
      castTo = static_cast<::onnx::TensorProto_DataType>(attributeProto.i());
    }
  }

  castParam->dstT = OnnxOpConverter::convertDataType(castTo);
  dstOp->main.value = castParam.release();
}

REGISTER_CONVERTER(CastOnnx, Cast);

}  // namespace converter
}  // namespace ace