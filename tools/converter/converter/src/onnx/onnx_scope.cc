#include <glog/logging.h>

#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

void OnnxScope::init() {
  const int initializerCount = graph_->initializer_size();
  for (int i = 0; i < initializerCount; ++i) {
    const auto& initializer = graph_->initializer(i);
    initializers.insert(std::make_pair(initializer.name(), &initializer));
    LOG(INFO) << "Initializer " << i << ": " << initializer.name();
  }

  const int inputCount = graph_->input_size();
  for (int i = 0; i < inputCount; ++i) {
    const auto& input = graph_->input(i);
    inputs.insert(std::make_pair(input.name(), &input));
    LOG(INFO) << "Input " << i << ": " << input.name();
  }

  const int outputCount = graph_->output_size();
  for (int i = 0; i < outputCount; ++i) {
    const auto& output = graph_->output(i);
    outputs.insert(std::make_pair(output.name(), &output));
    LOG(INFO) << "Output " << i << ": " << output.name();
  }
}

int OnnxScope::lookupTensor(std::string name) {
  // onnx have optional input, which may be a placeholder when pytorch export
  // onnx model, so drop this input, but we should check it out sometimes.
  if (name == "") {
    return -1;
  }

  const auto iter = mTensorIdx.find(name);
  if (iter != mTensorIdx.end()) {
    return iter->second;
  }
  return -1;
}

std::pair<int, int> OnnxScope::buildTensorArrayOp(
    std::vector<int> element_shape, bool identical, const std::string& name) {
  std::unique_ptr<ace::OpT> tensorArrayOp(new ace::OpT);
  tensorArrayOp->name = name;
  tensorArrayOp->type = ace::OpType_TensorArray;
  tensorArrayOp->defaultDimentionFormat = DataFormat_NCHW;
  tensorArrayOp->main.type = ace::OpParameter_TensorArray;
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = DataType_DT_FLOAT;
  tensorArray->dynamic_size = true;
  tensorArray->identical_element_shapes = identical;
  tensorArray->element_shape = element_shape;
  tensorArrayOp->main.value = tensorArray;
  tensorArrayOp->inputIndexes.push_back(
      buildIntConstOp({1}, name + "/init_size"));
  int idx_handle = declareTensor(name + "/handle");
  int idx = declareTensor(name);
  tensorArrayOp->outputIndexes.push_back(idx_handle);
  tensorArrayOp->outputIndexes.push_back(idx);
  oplists().emplace_back(std::move(tensorArrayOp));
  return std::make_pair(idx_handle, idx);
}

void OnnxScope::buildAccumulate(const std::string& name,
                                const std::string& uName,
                                const std::string& iName,
                                const std::string& oName) {
  // for while_body: %user_defined_val = Add(%user_defined_val, %output)
  int idxAcc = declareTensor(name + "/accumulate_u");
  ace::OpT* accumulateOp = new ace::OpT;
  accumulateOp->name = name + "/accumulate";
  accumulateOp->type = ace::OpType_TensorArrayWrite;
  accumulateOp->defaultDimentionFormat = DataFormat_NCHW;
  accumulateOp->main.type = ace::OpParameter_TensorArray;
  auto param = new ace::TensorArrayT;
  param->T = ace::DataType_DT_FLOAT;
  accumulateOp->main.value = param;
  // handle, index, value, flow_in
  addInputForOp(accumulateOp, uName + "/handle");
  addInputForOp(accumulateOp, iName);
  addInputForOp(accumulateOp, oName);
  addInputForOp(accumulateOp, uName);
  accumulateOp->outputIndexes.push_back(idxAcc);
  oplists().emplace_back(accumulateOp);
  mSubNet->outputs.push_back(idxAcc);
}

std::vector<std::string> OnnxScope::buildSubGraph(const onnx::GraphProto* graph,
                                                  std::string& name,
                                                  bool forLoop) {
  std::unique_ptr<ace::SubGraphProtoT> subgraph(new ace::SubGraphProtoT);
  subgraph->name = name;
  std::unique_ptr<OnnxScope> scope(
      new OnnxScope(graph, subgraph.get(), mNet, this));
  const auto& initializers = scope->initializers;
  const auto& inputs = scope->inputs;
  const auto& outputs = scope->outputs;
  // set input node to ace net
  for (int index = 0; index < graph->input_size(); ++index) {
    auto inputName = graph->input(index).name();
    bool notHaveInitializer =
        initializers.find(inputName) == initializers.end();
    if (notHaveInitializer) {
      ace::OpT* aceOp = new ace::OpT;
      aceOp->name = inputName;
      aceOp->type = ace::OpType_Input;
      aceOp->main.type = ace::OpParameter_Input;
      auto inputParam = new ace::InputT;
      const auto it = inputs.find(inputName);
      const auto& tensorInfo = (it->second)->type().tensor_type();
      const int inputDimSize = tensorInfo.shape().dim_size();
      inputParam->dims.resize(inputDimSize);
      for (int i = 0; i < inputDimSize; ++i) {
        inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
      }
      inputParam->dtype =
          OnnxOpConverter::convertDataType(tensorInfo.elem_type());
      inputParam->dformat = ace::DataFormat_NCHW;
      aceOp->outputIndexes.push_back(scope->declareTensor(inputName));
      aceOp->main.value = inputParam;
      subgraph->inputs.emplace_back(aceOp->outputIndexes[0]);
      subgraph->nodes.emplace_back(aceOp);
    }
  }
  // Find Extra Input from outside graph
  std::map<std::string, int> outsideInputs;
  for (int i = 0; i < graph->node_size(); i++) {
    const auto& onnxNode = graph->node(i);
    const auto& opType = onnxNode.op_type();
    // name maybe null, use the first output name as node-name
    const auto& name = onnxNode.output(0);
    auto opConverter = OnnxOpConverterSuit::get()->search(opType);
    ace::OpT* aceOp = new ace::OpT;
    aceOp->name = name;
    aceOp->type = opConverter->opType();
    aceOp->main.type = opConverter->type();
    for (int k = 0; k < onnxNode.input_size(); ++k) {
      const auto& inputName = onnxNode.input(k);
      if (scope->lookupTensor(inputName) >= 0) {
        continue;
      }
      // onnx subgraph may use tensor from initializers in outter level graph,
      // recurrsive find it
      for (auto curScope = scope.get(); curScope != nullptr;) {
        const auto& curInits = curScope->initializers;
        const auto it = curInits.find(inputName);
        if (it != curInits.end()) {
          // Create const Op
          ace::OpT* constOp = new ace::OpT;
          constOp->type = ace::OpType_Const;
          constOp->main.type = ace::OpParameter_Blob;
          constOp->main.value =
              OnnxOpConverter::convertTensorToBlob(it->second);
          constOp->name = it->first;
          constOp->outputIndexes.push_back(scope->declareTensor(it->first));
          subgraph->nodes.emplace_back(constOp);
          break;
        }
        curScope = reinterpret_cast<decltype(curScope)>(curScope->mParent);
      }
    }
    // build input and output
    for (int k = 0; k < onnxNode.input_size(); k++) {
      auto inputName = onnxNode.input(k);
      int idx = scope->lookupTensor(inputName);
      if (idx < 0) {
        auto iter = outsideInputs.find(inputName);
        if (iter == outsideInputs.end()) {
          idx = scope->declareTensor(inputName);
          std::unique_ptr<ace::OpT> inputOp(new ace::OpT);
          inputOp->name = inputName;
          inputOp->type = ace::OpType_Input;
          inputOp->main.type = ace::OpParameter_Input;
          auto param = new ace::InputT;
          param->dtype = ace::DataType_DT_INT32;
          param->dformat = ace::DataFormat_NCHW;
          inputOp->main.value = param;
          inputOp->outputIndexes.push_back(idx);
          subgraph->nodes.emplace_back(std::move(inputOp));
          outsideInputs.insert(std::make_pair(inputName, idx));
        } else {
          idx = iter->second;
        }
      }
      aceOp->inputIndexes.push_back(idx);
    }
    for (int k = 0; k < onnxNode.output_size(); k++) {
      aceOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
    }
    auto originIdx = subgraph->inputs.size();
    opConverter->run(aceOp, &onnxNode, scope.get());
    // subgraph own by op may introduce extra input which is not exist on
    // current graph, create it in op converter and detect it by
    // subgraph->inputs
    for (int inputIdx = originIdx; inputIdx < subgraph->inputs.size();
         ++inputIdx) {
      auto idx = subgraph->inputs[inputIdx];
      outsideInputs.insert(std::make_pair(scope->lookupTensorByIdx(idx), idx));
    }
    subgraph->inputs.erase(subgraph->inputs.begin() + originIdx,
                           subgraph->inputs.end());
    subgraph->nodes.emplace_back(aceOp);
  }
  if (!forLoop) {
    std::vector<std::string> resOutside;
    for (auto& iter : outsideInputs) {
      subgraph->inputs.emplace_back(iter.second);
      resOutside.emplace_back(iter.first);
    }
    for (int i = 0; i < graph->output_size(); ++i) {
      int idx = scope->lookupTensor(graph->output(i).name());
      // ACE_ASSERT(idx >= 0);
      if (idx >= 0) {
        subgraph->outputs.push_back(idx);
      }
    }
    mNet->subgraphs.emplace_back(std::move(subgraph));
    return resOutside;
  }
  int N = graph->input_size() - 2, K = graph->output_size() - N - 1;
  for (int i = 0; i < N + 1; i++) {
    int idx = scope->lookupTensor(graph->output(i).name());
    // ace_ASSERT(idx >= 0);
    if (idx >= 0) {
      subgraph->outputs.push_back(idx);
    }
  }
  std::vector<std::string> resOutside;
  for (auto& iter : outsideInputs) {
    subgraph->inputs.emplace_back(iter.second);
    subgraph->outputs.emplace_back(iter.second);
    resOutside.emplace_back(iter.first);
  }
  for (int i = 0; i < K; ++i) {
    int idx = scope->lookupTensor(graph->output(i + N + 1).name());
    // ace_ASSERT(idx >= 0);
    if (idx >= 0) {
      subgraph->outputs.push_back(idx);
    }
  }
  mNet->subgraphs.emplace_back(std::move(subgraph));
  return resOutside;
}
}  // namespace converter
}  // namespace ace