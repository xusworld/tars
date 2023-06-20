#pragma once

#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_scope.h"
#include "src/onnx/utils.h"

namespace ace {
namespace converter {

class DefaultOnnxOpConverter : public OnnxOpConverter {
 public:
  virtual void run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) override;

  virtual ace::OpParameter type() override { return OpParameter_Extra; }
  virtual ace::OpType opType() override { return OpType_Extra; }
};

}  // namespace converter
}  // namespace ace