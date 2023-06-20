#pragma once

#include <map>
#include <string>
#include <vector>

#include "ace/schema/ace_generated.h"
#include "src/onnx/converter_scope.h"
#include "src/onnx/proto/onnx.pb.h"

namespace ace {
namespace converter {

class OnnxScope : public ConverterScope {
 public:
  OnnxScope(const onnx::GraphProto* graph, ace::NetT* net)
      : graph_(graph), ConverterScope(net) {
    init();
  }

  OnnxScope(const onnx::GraphProto* graph, ace::SubGraphProtoT* subnet,
            ace::NetT* net, OnnxScope* parent)
      : graph_(graph), ConverterScope(subnet, net, parent) {
    init();
  }

  std::pair<int, int> buildTensorArrayOp(std::vector<int> element_shape,
                                         bool identical,
                                         const std::string& name);

  void buildAccumulate(const std::string& name, const std::string& uName,
                       const std::string& iName, const std::string& oName);

  // Return extra input needed from subgraph
  // WhileModule implemention acquire
  std::vector<std::string> buildSubGraph(const onnx::GraphProto* graph,
                                         std::string& name, bool forLoop);

 public:
  virtual int lookupTensor(std::string name);
  // Use to save onnx model inputs and output information.
  std::map<std::string, const onnx::TensorProto*> initializers;
  std::map<std::string, const onnx::ValueInfoProto*> inputs;
  std::map<std::string, const onnx::ValueInfoProto*> outputs;

 private:
  // Setup basic model information.
  void init();
  // Onnx Graph.
  const onnx::GraphProto* graph_;
};

}  // namespace converter
}  // namespace ace