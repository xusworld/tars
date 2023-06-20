#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(GridSampleOnnx);

ace::OpType GridSampleOnnx::opType() { return ace::OpType_GridSample; }

ace::OpParameter GridSampleOnnx::type() { return ace::OpParameter_GridSample; }

void GridSampleOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         OnnxScope *scope) {
  auto gridSampleParam = new ace::GridSampleT;

  gridSampleParam->mode = ace::SampleMode_BILINEAR;
  gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
  gridSampleParam->alignCorners = false;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto &attributeProto = onnxNode->attribute(i);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->mode = ace::SampleMode_BILINEAR;
          break;
        case 1:
          gridSampleParam->mode = ace::SampleMode_NEAREST;
          break;
        default:
          LOG(FATAL) << "Unknown mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "padding_mode") {
      switch (attributeProto.i()) {
        case 0:
          gridSampleParam->paddingMode = ace::BorderMode_ZEROS;
          break;
        case 1:
          gridSampleParam->paddingMode = ace::BorderMode_CLAMP;
          break;
        case 2:
          gridSampleParam->paddingMode = ace::BorderMode_REFLECTION;
          break;
        default:
          LOG(FATAL) << "Unknown padding mode for " << onnxNode->name() << "!";
          break;
      }
    }
    if (attributeName == "align_corners") {
      gridSampleParam->alignCorners = attributeProto.i();
    }
  }

  dstOp->main.value = gridSampleParam;
}

// REGISTER_CONVERTER(GridSampleOnnx, GridSample);

// When we export torch.nn.functional.grid_sample to onnx, it's called
// GridSampler rather than GridSample, thus, we have to add the "r"
#define REGISTER_CONVERTER_r(name, opType) \
  static OnnxOpConverterRegister<name> _Convert_##opType(#opType "r")
REGISTER_CONVERTER_r(GridSampleOnnx, GridSample);

}  // namespace converter
}  // namespace ace