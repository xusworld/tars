#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(EluOnnx);
DECLARE_OP_CONVERTER(SEluOnnx);

ace::OpType EluOnnx::opType() { return ace::OpType_ELU; }
ace::OpType SEluOnnx::opType() { return ace::OpType_Selu; }

ace::OpParameter EluOnnx::type() { return ace::OpParameter_ELU; }
ace::OpParameter SEluOnnx::type() { return ace::OpParameter_Selu; }

void EluOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                  OnnxScope *scope) {
  auto eluParam = new ace::ELUT;

  float alpha = 1.0f;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "alpha") {
      alpha = attributeProto.f();
    }
  }

  eluParam->alpha = alpha;

  dstOp->main.value = eluParam;
}
void SEluOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  auto seluParam = new ace::SeluT;

  float alpha = 1.67326, gamma = 1.0507;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "alpha") {
      alpha = attributeProto.f();
    } else if (attributeName == "gamma") {
      gamma = attributeProto.f();
    }
  }

  seluParam->alpha = alpha;
  seluParam->scale = gamma;

  dstOp->main.value = seluParam;
}

REGISTER_CONVERTER(EluOnnx, Elu);
REGISTER_CONVERTER(SEluOnnx, Selu);

}  // namespace converter
}  // namespace ace