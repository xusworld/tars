#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ArgMaxOnnx);
DECLARE_OP_CONVERTER(ArgMinOnnx);

ace::OpType ArgMaxOnnx::opType() { return ace::OpType_ArgMax; }

ace::OpParameter ArgMaxOnnx::type() { return ace::OpParameter_ArgMax; }

ace::OpType ArgMinOnnx::opType() { return ace::OpType_ArgMin; }

ace::OpParameter ArgMinOnnx::type() { return ace::OpParameter_ArgMax; }

static void _run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                 OnnxScope *scope) {
  auto axisT = new ace::ArgMaxT;
  int axis = 0;
  int keepdims = 1;
  int selectLastIndex = 0;  // Boolean value. Default to False.

  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();

    if (attributeName == "axis") {
      axis = attributeProto.i();
    }
    if (attributeName == "keepdims") {
      keepdims = attributeProto.i();
    }
    if (attributeName == "select_last_index") {
      // Ignored for now. ace argmax implementation does not support this yet.
      selectLastIndex = attributeProto.i();
    }
  }
  axisT->axis = axis;
  axisT->topK = 1;
  axisT->outMaxVal = 0;
  if (keepdims == 1) {
    std::unique_ptr<ace::OpT> op(new ace::OpT);
    op->name = dstOp->name + "/not_keepdim";
    op->type = dstOp->type;
    op->main.type = dstOp->main.type;
    op->main.value = axisT;
    op->inputIndexes = dstOp->inputIndexes;
    std::vector<int> midIndexs(1, scope->declareTensor(op->name));
    op->outputIndexes = dstOp->inputIndexes = midIndexs;
    dstOp->type = ace::OpType_Unsqueeze;
    auto param = new ace::SqueezeParamT;
    param->squeezeDims.assign({axis});
    dstOp->main.type = ace::OpParameter_SqueezeParam;
    dstOp->main.value = param;
    scope->oplists().emplace_back(std::move(op));
    return;
  }
  dstOp->main.value = axisT;
}

void ArgMaxOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope *scope) {
  _run(dstOp, onnxNode, scope);
}

void ArgMinOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope *scope) {
  _run(dstOp, onnxNode, scope);
}

REGISTER_CONVERTER(ArgMaxOnnx, ArgMax);
REGISTER_CONVERTER(ArgMinOnnx, ArgMin);

}  // namespace converter
}  // namespace ace