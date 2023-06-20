#include <glog/logging.h>

#include <cassert>

#include "src/onnx/onnx_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/utils.h"

namespace ace {
namespace converter {

int Onnx2AceNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<ace::NetT>& netT) {
  LOG(INFO) << "Onnx2AceNet|Processing onnx model " << inputModel;
  std::string modelDir;
  size_t pos = inputModel.find_last_of("\\/");
  if (pos != std::string::npos) {
    modelDir = inputModel.substr(0, pos + 1);
  }

  onnx::ModelProto onnxModel;
  // read ONNX Model
  bool success = OnnxReadProtoFromBinary(inputModel.c_str(), &onnxModel);
  CHECK(success) << "read onnx model failed: " << inputModel;
  if (!success) {
    LOG(FATAL) << "[ERROR] Model file is not onnx model.\n";
    return 1;
  }

  LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();

  const auto& onnxGraph = onnxModel.graph();
  const int nodeCount = onnxGraph.node_size();

  LOG(INFO) << "Total Graph Node count: " << nodeCount;
  std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get()));
  // find the inputs which do not have initializer
  const auto& initializers = scope->initializers;
  const auto& inputs = scope->inputs;
  const auto& outputs = scope->outputs;

  LOG(INFO) << "Graph initializers size: " << initializers.size();
  LOG(INFO) << "Graph inputs size: " << inputs.size();
  LOG(INFO) << "Graph outputs size: " << outputs.size();

  LOG(INFO) << "Processing model inputs ...";
  for (const auto& iter : inputs) {
    bool notHaveInitializer =
        initializers.find(iter.first) == initializers.end();
    if (notHaveInitializer) {
      ace::OpT* aceOp = new ace::OpT;
      aceOp->name = iter.first;
      aceOp->type = ace::OpType_Input;
      aceOp->main.type = ace::OpParameter_Input;

      LOG(INFO) << "name: " << aceOp->name << " type: " << aceOp->type;

      auto inputParam = new ace::InputT;
      const auto it = inputs.find(iter.first);
      DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
      const auto& tensorInfo = (it->second)->type().tensor_type();

      const int inputDimSize = tensorInfo.shape().dim_size();
      inputParam->dims.resize(inputDimSize);
      for (int i = 0; i < inputDimSize; ++i) {
        inputParam->dims[i] = tensorInfo.shape().dim(i).dim_value();
        LOG(INFO) << "inputParam->dims: " << inputParam->dims[i];
      }

      inputParam->dtype =
          OnnxOpConverter::convertDataType(tensorInfo.elem_type());
      inputParam->dformat = ace::DataFormat_NCHW;
      aceOp->outputIndexes.push_back(scope->declareTensor(iter.first));
      aceOp->main.value = inputParam;
      netT->oplists.emplace_back(aceOp);
    }
  }

  LOG(INFO) << "Processing model nodes ...";
  for (int i = 0; i < nodeCount; ++i) {
    const auto& onnxNode = onnxGraph.node(i);
    const auto& opType = onnxNode.op_type();

    LOG(INFO) << "ONNX op type: " << opType;
    // name maybe null, use the first output name as node-name
    const auto& name = onnxNode.output(0);
    auto opConverter = OnnxOpConverterSuit::get()->search(opType);
    assert(opConverter != nullptr);

    ace::OpT* aceOp = new ace::OpT;
    aceOp->name = name;
    aceOp->type = opConverter->opType();
    aceOp->main.type = opConverter->type();

    LOG(INFO) << "op->name: " << aceOp->name << " op->type: " << aceOp->type;

    // convert initializer to be Constant node(op)
    for (int k = 0; k < onnxNode.input_size(); ++k) {
      const auto& inputName = onnxNode.input(k);
      const auto it = initializers.find(inputName);
      if (it != initializers.end() && scope->lookupTensor(it->first) == -1) {
        // Create const Op
        ace::OpT* constOp = new ace::OpT;
        constOp->type = ace::OpType_Const;
        constOp->main.type = ace::OpParameter_Blob;
        constOp->main.value =
            OnnxOpConverter::convertTensorToBlob(it->second, modelDir);
        constOp->name = it->first;
        constOp->outputIndexes.push_back(scope->declareTensor(it->first));
        netT->oplists.emplace_back(constOp);
      }
    }
    // build input and output
    for (int k = 0; k < onnxNode.input_size(); k++) {
      int inputIdx = scope->lookupTensor(onnxNode.input(k));
      if (inputIdx < 0) {
        LOG(INFO) << "Check it out ==> " << aceOp->name
                  << " has empty input, the index is " << k;
      }
      aceOp->inputIndexes.push_back(inputIdx);
    }
    for (int k = onnxNode.input_size() - 1;
         k >= 0 && aceOp->inputIndexes[k] < 0; --k) {
      aceOp->inputIndexes.pop_back();
    }
    for (int k = 0; k < onnxNode.output_size(); k++) {
      aceOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
    }
    // build op
    opConverter->run(aceOp, &onnxNode, scope.get());
    netT->oplists.emplace_back(aceOp);
  }

  netT->tensorNumber = netT->tensorName.size();
  LOG(INFO) << "tensor name: " << netT->tensorNumber;

  // set ace net output name
  LOG(INFO) << "Processing model outputs ...";
  for (const auto& iter : outputs) {
    netT->outputName.push_back(iter.first);
  }

  netT->sourceType = ace::NetSource_ONNX;
  netT->bizCode = bizCode;

  LOG(INFO) << "Success...";
  return 0;
}

}  // namespace converter
}  // namespace ace
