#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ReduceOnnx);

ace::OpType ReduceOnnx::opType() { return ace::OpType_Reduction; }
ace::OpParameter ReduceOnnx::type() { return ace::OpParameter_ReductionParam; }

void ReduceOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope *scope) {
  auto param = new ace::ReductionParamT;

  std::vector<int> axes;
  bool keepdims = true;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();

    if (attributeName == "axes") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      const int size = attributeProto.ints_size();
      for (int k = 0; k < size; ++k) {
        axes.push_back(attributeProto.ints(k));
      }
    } else if (attributeName == "keepdims") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      keepdims = static_cast<bool>(attributeProto.i());
    }
  }

  auto type = onnxNode->op_type();
  if (type == "ReduceMean") {
    param->operation = ace::ReductionType_MEAN;
  } else if (type == "ReduceMax") {
    param->operation = ace::ReductionType_MAXIMUM;
  } else if (type == "ReduceMin") {
    param->operation = ace::ReductionType_MINIMUM;
  } else if (type == "ReduceProd") {
    param->operation = ace::ReductionType_PROD;
  } else if (type == "ReduceSum") {
    param->operation = ace::ReductionType_SUM;
  } else {
    DLOG(ERROR) << "TODO ==> " << type;
  }

  param->dType = ace::DataType_DT_FLOAT;
  param->dim = axes;
  param->keepDims = keepdims;
  dstOp->main.value = param;
}

REGISTER_CONVERTER(ReduceOnnx, ReduceMean);
REGISTER_CONVERTER(ReduceOnnx, ReduceMax);
REGISTER_CONVERTER(ReduceOnnx, ReduceMin);
REGISTER_CONVERTER(ReduceOnnx, ReduceProd);
REGISTER_CONVERTER(ReduceOnnx, ReduceSum);

}  // namespace converter
}  // namespace ace