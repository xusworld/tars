#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(GridSampleOnnxClassic);

ace::OpType GridSampleOnnxClassic::opType() { return ace::OpType_GridSample; }

ace::OpParameter GridSampleOnnxClassic::type() {
  return ace::OpParameter_GridSample;
}

void GridSampleOnnxClassic::run(ace::OpT *dstOp,
                                const onnx::NodeProto *onnxNode,
                                OnnxScope *scope) {
  auto gridSampleParam = new ace::GridSampleT;

  gridSampleParam->mode = ace::SampleMode_BILINEAR;
  gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
  gridSampleParam->alignCorners = false;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "mode") {
      gridSampleParam->mode = ace::SampleMode_BILINEAR;
      if (attributeProto.s() == "bilinear") {
        gridSampleParam->mode = ace::SampleMode_BILINEAR;
      } else if (attributeProto.s() == "nearest") {
        gridSampleParam->mode = ace::SampleMode_NEAREST;
      } else {
        // LOG_INFO.stream() << "Don't support mode " << attributeProto.s();
      }
    }
    if (attributeName == "padding_mode") {
      gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
      if (attributeProto.s() == "zeros") {
        gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
      } else if (attributeProto.s() == "border") {
        gridSampleParam->paddingMode = ace::BorderMode_CLAMP;
      } else if (attributeProto.s() == "reflection") {
        gridSampleParam->paddingMode = ace::BorderMode_REFLECTION;
      } else {
        // LOG_INFO.stream() << "Don't support padding_mode "
        // << attributeProto.s();
      }
    }
    if (attributeName == "align_corners") {
      gridSampleParam->alignCorners = attributeProto.i();
    }
  }

  dstOp->main.value = gridSampleParam;
}

REGISTER_CONVERTER(GridSampleOnnxClassic, GridSample);

}  // namespace converter
}  // namespace ace