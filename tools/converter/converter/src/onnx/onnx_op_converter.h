#pragma once

#include <map>
#include <string>
#include <vector>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_scope.h"
#include "src/onnx/proto/onnx.pb.h"

namespace ace {
namespace converter {

class OnnxOpConverter {
 public:
  OnnxOpConverter() = default;
  virtual ~OnnxOpConverter() = default;
  // do op converter
  virtual void run(ace::OpT* dstOp, const onnx::NodeProto* node,
                   OnnxScope* scope) = 0;
  // op attrs
  virtual ace::OpParameter type() = 0;
  virtual ace::OpType opType() = 0;

  static ace::DataType convertDataType(int32_t type);
  static ace::BlobT* convertTensorToBlob(const onnx::TensorProto* tensor,
                                         const std::string& modelDir = "");

  // static std::unique_ptr<ace::SubGraphProtoT> buildSubGraph(const
  // onnx::GraphProto* graph, std::string& name);
 protected:
  std::vector<std::unique_ptr<ace::SubGraphProtoT>> subgraphs_;
};

}  // namespace converter
}  // namespace ace