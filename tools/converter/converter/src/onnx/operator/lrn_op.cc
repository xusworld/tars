#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(LRNOnnx);

ace::OpType LRNOnnx::opType() { return ace::OpType_LRN; }

ace::OpParameter LRNOnnx::type() { return ace::OpParameter_LRN; }

void LRNOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                  OnnxScope* scope) {
  auto param = new ace::LRNT;

  int size = 0;
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "size") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      size = attributeProto.i();
    } else if (attributeName == "alpha") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      alpha = attributeProto.f();
    } else if (attributeName == "beta") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      beta = attributeProto.f();
    } else if (attributeName == "bias") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      bias = attributeProto.f();
    }
  }

  param->alpha = alpha;
  param->beta = beta;
  param->localSize = size;
  param->regionType = 0;
  param->bias = bias;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(LRNOnnx, LRN);

}  // namespace converter
}  // namespace ace