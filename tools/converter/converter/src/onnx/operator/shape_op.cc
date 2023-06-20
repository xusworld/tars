#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ShapeOnnx);

ace::OpType ShapeOnnx::opType() { return ace::OpType_Shape; }
ace::OpParameter ShapeOnnx::type() { return ace::OpParameter_NONE; }

void ShapeOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                    OnnxScope* scope) {
  dstOp->defaultDimentionFormat = ace::DataFormat_NCHW;
}

REGISTER_CONVERTER(ShapeOnnx, Shape);

DECLARE_OP_CONVERTER(SizeOnnx);

ace::OpType SizeOnnx::opType() { return ace::OpType_Size; }
ace::OpParameter SizeOnnx::type() { return ace::OpParameter_NONE; }

void SizeOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  dstOp->defaultDimentionFormat = ace::DataFormat_NCHW;
}

REGISTER_CONVERTER(SizeOnnx, Size);

}  // namespace converter
}  // namespace ace