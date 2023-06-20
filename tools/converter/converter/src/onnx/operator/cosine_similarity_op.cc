#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(CosineSimilarityOnnx);

ace::OpType CosineSimilarityOnnx::opType() {
  return ace::OpType_CosineSimilarity;
}

ace::OpParameter CosineSimilarityOnnx::type() { return ace::OpParameter_NONE; }

void CosineSimilarityOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                               OnnxScope *scope) {
  std::string type;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    auto att = onnxNode->attribute(i);
    if ("operator" == att.name()) {
      type = att.s();
      break;
    }
  }
  DCHECK(type == "cosine_similarity") << " NOT SUPPPRT";
  return;
}

REGISTER_CONVERTER(CosineSimilarityOnnx, ATen);

}  // namespace converter
}  // namespace ace