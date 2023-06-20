#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"
namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ROIPoolingOnnx);

ace::OpType ROIPoolingOnnx::opType() { return ace::OpType_ROIPooling; }

ace::OpParameter ROIPoolingOnnx::type() {
  return ace::OpParameter_RoiParameters;
}

void ROIPoolingOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         OnnxScope *scope) {
  auto roiPool = new ace::RoiParametersT;
  roiPool->spatialScale = 1;
  roiPool->poolType = ace::PoolType_MAXPOOL;

  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "pooled_shape") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS)
          << "Node Attribute ERROR";
      DCHECK(attributeProto.ints_size() == 2) << "Node Attribute ERROR";
      roiPool->pooledHeight = attributeProto.ints(0);
      roiPool->pooledWidth = attributeProto.ints(1);
    } else if (attributeName == "spatial_scale") {
      DCHECK(attributeProto.type() ==
             ::onnx::AttributeProto_AttributeType_FLOAT)
          << "Node Attribute ERROR";
      roiPool->spatialScale = attributeProto.f();
    } else {
      DLOG(ERROR) << "TODO!";
    }
  }

  dstOp->main.value = roiPool;
};

REGISTER_CONVERTER(ROIPoolingOnnx, MaxRoiPool);

}  // namespace converter
}  // namespace ace