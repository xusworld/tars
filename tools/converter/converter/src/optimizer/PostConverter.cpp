#include <glog/logging.h>

#include <cassert>
#include <set>
#include <unordered_set>

#include "GenerateSubGraph.hpp"
#include "PostConverter.hpp"
#include "PostTreatUtils.hpp"
#include "Program.hpp"
#include "SubGraphComplete.hpp"
#include "TemplateMerge.hpp"
#include "express/Optimizer.hpp"

namespace ace {
namespace Express {
static std::vector<int> NetInputIndices(const ace::NetT* net) {
  std::vector<int> input_indices;
  for (const auto& op : net->oplists) {
    if (op->type == ace::OpType_Input) {
      const auto& indices = op->outputIndexes;
      input_indices.insert(input_indices.end(), indices.begin(), indices.end());
    }
  }
  return std::move(input_indices);
}

SubGraphProtoT* FindSubGraphByName(
    const std::vector<SubGraphProtoT*>& subgraphs,
    const std::string& subgraph_name) {
  for (SubGraphProtoT* subgraph : subgraphs) {
    if (subgraph->name == subgraph_name) {
      return subgraph;
    }
  }
  return nullptr;
}

bool CompleteSubGraph(const std::unordered_map<std::string, VARP>& inputs,
                      const SubGraphProtoT* subgraph) {
  auto* ctx = Global<OptimizeContext>::Get();
  assert(ctx != nullptr);
  // Disable verbose for subgraph.
  bool verbose = ctx->verbose;
  ctx->verbose = false;
  std::vector<std::string> outputNames;
  for (auto o : subgraph->outputs) {
    outputNames.emplace_back(subgraph->tensors[o]);
  }

  SubGraphProtoT* mutable_subgraph =  // NOLINT
      FindSubGraphByName(ctx->subgraphs, subgraph->name);
  assert(mutable_subgraph == subgraph);
  std::unique_ptr<ace::NetT> subnet(new ace::NetT);
  subnet->oplists = std::move(mutable_subgraph->nodes);
  subnet->tensorName = mutable_subgraph->tensors;
  subnet->sourceType = ctx->source;
  subnet->outputName = outputNames;

  std::unique_ptr<ace::NetT> new_subnet = ctx->RunOptimize(subnet, inputs);
  mutable_subgraph->nodes = std::move(subnet->oplists);

  ace::SubGraphProtoT* new_subgraph(new ace::SubGraphProtoT);
  new_subgraph->name = mutable_subgraph->name;
  new_subgraph->inputs = NetInputIndices(new_subnet.get());
  new_subgraph->outputs.clear();
  for (auto& output : outputNames) {
    for (int i = 0; i < new_subnet->tensorName.size(); ++i) {
      if (new_subnet->tensorName[i] == output) {
        new_subgraph->outputs.emplace_back(i);
        break;
      }
    }
  }
  assert(new_subgraph->outputs.size() == outputNames.size());
  new_subgraph->nodes = std::move(new_subnet->oplists);
  new_subgraph->tensors = new_subnet->tensorName;

  assert(!FindSubGraphByName(ctx->completed_subgraphs, new_subgraph->name));
  ctx->completed_subgraphs.push_back(new_subgraph);

  // Recovery verbose.
  ctx->verbose = verbose;
  return true;
}

void RunNetPass(const std::vector<std::string>& passes,
                std::unique_ptr<ace::NetT>& originNet) {
  for (auto pass : passes) {
    auto convert = PostConverter::get(pass);
    if (nullptr == convert) {
      LOG(INFO) << "Can't find pass of " << pass << "\n";
      continue;
    }
    bool valid = convert->onExecute(originNet);
    if (!valid) {
      LOG(INFO) << "Run " << pass << "Error\n";
    }
  }
}

std::unique_ptr<ace::NetT> RunExtraPass(
    std::unique_ptr<ace::NetT>& originNet,
    const std::unordered_map<std::string, VARP>& inputs) {
  auto program = ace::Express::Program::create(originNet.get(), true);
  program->input(inputs);

  std::string pass = "TFExtra";
  switch (originNet->sourceType) {
    case ace::NetSource_TFLITE:
      pass = "TFliteExtra";
      break;
    case ace::NetSource_TENSORFLOW:
      pass = "TFExtra";
      break;
    case ace::NetSource_CAFFE:
      pass = "CaffeExtra";
      break;
    case ace::NetSource_ONNX:
      pass = "OnnxExtra";
      break;
    case ace::NetSource_TORCH:
      pass = "TorchExtra";
      break;
    default:
      break;
  }
  auto& merge = ace::Express::TemplateMerge::getInstance(pass);
  merge.onExecute(program->outputs());

  std::unique_ptr<ace::NetT> newNet(new ace::NetT);
  auto outputs = program->outputs();
  newNet->sourceType = originNet->sourceType;
  newNet->bizCode = originNet->bizCode;
  newNet->outputName = originNet->outputName;
  Variable::save(outputs, newNet.get());
  return std::move(newNet);
}

std::unique_ptr<ace::NetT> RunMergePass(
    std::unique_ptr<ace::NetT>& originNet,
    const std::unordered_map<std::string, VARP>& inputs,
    PassPriority priority) {
  auto program = ace::Express::Program::create(originNet.get(), true);
  program->input(inputs);

  std::string pass = "Merge";
  auto& merge = ace::Express::TemplateMerge::getInstance(pass);
  merge.onExecute(program->outputs(), priority);

  std::unique_ptr<ace::NetT> newNet(new ace::NetT);
  auto outputs = program->outputs();
  newNet->sourceType = originNet->sourceType;
  newNet->bizCode = originNet->bizCode;
  newNet->outputName = originNet->outputName;
  Variable::save(outputs, newNet.get());

  RunNetPass({"RemoveUnusefulOp"}, newNet);
  return std::move(newNet);
}

std::unique_ptr<ace::NetT> optimizeNetImpl(
    std::unique_ptr<ace::NetT>& originNet,
    const std::unordered_map<std::string, VARP>& inputs) {
  auto* ctx = Global<OptimizeContext>::Get();
  assert(ctx != nullptr);

  if (ctx->is_training) {
    LOG(INFO) << "convert model for training, reserve BatchNorm and Dropout";
  }
  if (originNet->oplists.size() <= 0) {
    return nullptr;
  }
  std::vector<std::string> postConvertPass;
  postConvertPass = {
      // Seperate Tensor for inplace op
      "RemoveInplace",

      // Remove Unuseful Op such as NoOp, Identity, Seq2Out,
      "RemoveUnusefulOp",

      // Remove Dropout, if `forTraining` flag is set, Dropout will be reserved
      "RemoveDropout",

      // Remove Dup op
      "FuseDupOp",

      // Turn InnerProduct from Caffe / Onnx to Convolution
      "TransformInnerProduct",

      // Turn Im2Seq from Caffe to Reshape
      "TransformIm2Seq",

      // Turn Caffe's ShuffleChannel to compose op
      "TransformShuffleChannel",

      // Turn Onnx's Pad to Tensorflow's Pad
      "TransformOnnxPad",
  };
  if (ctx->is_training) {
    std::vector<std::string>::iterator iter;
    for (iter = postConvertPass.begin(); iter != postConvertPass.end();
         iter++) {
      if (*iter == "RemoveDropout") {
        postConvertPass.erase(iter);
      }
    }
  }
  RunNetPass(postConvertPass, originNet);

  std::unique_ptr<ace::NetT> newNet;
  newNet = std::move(RunExtraPass(originNet, inputs));

  newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_HIGH));

  std::vector<std::string> afterProgramConvert = {
      // Turn BatchNormal to Scale When inference, if `forTraining` flag is set,
      // BN will be reserved
      "TransformBatchNormal",

      // expand ShapeN to N Shapes
      "ResolveTfShapeN",

      // WARNNING: should merge BN and Scale before Relu and Relu6

      // Merge BN info Convolution, if `forTraining` flag is set, BN will be
      // reserved
      "MergeBNToConvolution",

      // Merge Scale info Convolution
      "MergeScaleToConvolution",

      // Merge Relu Convolution
      "MergeReluToConvolution",

      // Merge Relu6 Convolution
      "MergeRelu6ToConvolution",

  };
  if (ctx->is_training) {
    std::vector<std::string>::iterator iter;
    for (iter = afterProgramConvert.begin(); iter != afterProgramConvert.end();
         iter++) {
      if (*iter == "TransformBatchNormal" || *iter == "MergeBNToConvolution") {
        afterProgramConvert.erase(iter);
      }
    }
  }
  RunNetPass(afterProgramConvert, newNet);

  newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_MIDDLE));

  afterProgramConvert = {
      // Add tensor dimension format convert for NC4HW4 - NHWC / NC4HW4 - NCHW
      "AddTensorFormatConverter",

      // Turn group convolution to Slice - Convolution - Concat
      "TransformGroupConvolution",

      // Remove output tensor convert
      "RemoveOutputTensorConvert",
  };
  RunNetPass(afterProgramConvert, newNet);

  // Maybe eliminate the redundant quantize and dequantize ops, then remove
  // the unuseful `Identity`.
  newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_LOW));

  // Maybe eliminate the redundant tensor format ops, then remove the unuseful
  // `Identity`.
  newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_LOW));
  newNet = std::move(RunMergePass(newNet, inputs, PASS_PRIORITY_FINAL));

  RunNetPass({"ReIndexTensor"}, newNet);

  return std::move(newNet);
}

bool fuseConstIntoSubgraph(ace::NetT* net,
                           const std::vector<ace::SubGraphProtoT*>& subgraphs) {
  if (subgraphs.empty()) {
    return false;
  }
  // Create Map for subGraphs
  // Key, protot, refcount
  std::map<std::string, std::pair<ace::SubGraphProtoT*, int>> subGraphMaps;
  std::set<ace::SubGraphProtoT*> modifiedSubGraph;
  for (auto s : subgraphs) {
    subGraphMaps.insert(std::make_pair(s->name, std::make_pair(s, 0)));
  }
  for (int i = 0; i < net->oplists.size(); ++i) {
    auto& op = net->oplists[i];
    if (op->type == ace::OpType_While) {
      auto param = op->main.AsWhileParam();
      subGraphMaps[param->body_graph].second++;
      subGraphMaps[param->cond_graph].second++;
      continue;
    }
    if (op->type == ace::OpType_If) {
      auto param = op->main.AsIfParam();
      subGraphMaps[param->else_graph].second++;
      subGraphMaps[param->then_graph].second++;
      continue;
    }
  }

  // Try Merge Const into subgraph
  // Search all const op
  std::vector<int> constOpIndexes(net->tensorName.size(), -1);
  for (int i = 0; i < net->oplists.size(); ++i) {
    auto& op = net->oplists[i];
    if (op->type == ace::OpType_Const) {
      constOpIndexes[op->outputIndexes[0]] = i;
    }
  }

  // Try Merge for while
  std::set<int> removeConstOpIndexes;
  for (int opIndex = 0; opIndex < net->oplists.size(); ++opIndex) {
    auto& op = net->oplists[opIndex];
    if (op->type != ace::OpType_While) {
      continue;
    }
    auto param = op->main.AsWhileParam();
    auto body = subGraphMaps[param->body_graph];
    auto cond = subGraphMaps[param->cond_graph];
    // Don't support for shared subgrah's optimize
    if (body.second > 1 || cond.second > 1) {
      continue;
    }
    assert(op->inputIndexes.size() == param->aliases_inputs.size());

    // Merge into subgraph
    std::set<int> removeInputs;
    std::set<int> bodyInputRemove;
    std::set<int> condInputRemove;
    auto mergeToSubGraph =
        [](ace::SubGraphProtoT* subGraph, std::set<int>& inputRemove,
           const ace::OpT* constOp, const std::string& inputName) {
          // Merge Const Index to Body
          for (auto& inputIndex : subGraph->inputs) {
            if (subGraph->tensors[inputIndex] == inputName) {
              inputRemove.insert(inputIndex);
              for (int v = 0; v < subGraph->nodes.size(); ++v) {
                auto& subOp = subGraph->nodes[v];
                if (subOp->type != ace::OpType_Input) {
                  continue;
                }
                if (subOp->outputIndexes[0] == inputIndex) {
                  auto src = constOp->main.AsBlob();
                  subOp->type = ace::OpType_Const;
                  subOp->main.type = ace::OpParameter_Blob;
                  subOp->main.value = new ace::BlobT;
                  *subOp->main.AsBlob() = *src;
                  break;
                }
              }
              break;
            }
          }
          return true;
        };
    for (int subI = 0; subI < op->inputIndexes.size(); ++subI) {
      auto index = op->inputIndexes[subI];
      auto constIndex = constOpIndexes[index];
      if (constIndex < 0) {
        continue;
      }
      // Don't support for graph shared input
      if (param->aliases_inputs[subI]->data.size() != 1) {
        continue;
      }
      auto inputName = param->aliases_inputs[subI]->data[0];
      // Don't support for const init and update next
      bool isUpdate = false;
      for (auto& update : param->aliases_updates) {
        for (auto updateName : update->data) {
          if (updateName == inputName) {
            isUpdate = true;
            break;
          }
        }
        if (isUpdate) {
          break;
        }
      }
      if (isUpdate) {
        continue;
      }
      // Count Refcount for const tensor
      int refCount = 0;
      for (int sub = constIndex + 1; sub < net->oplists.size(); ++sub) {
        auto& subOp = net->oplists[sub];
        for (auto subIndex : subOp->inputIndexes) {
          if (subIndex == index) {
            refCount++;
            break;
          }
        }
      }
      if (refCount > 1) {
        // The const input is shared with other op
        continue;
      }
      auto& constOp = net->oplists[constIndex];
      assert(constOp->main.type == ace::OpParameter_Blob);

      removeConstOpIndexes.insert(constIndex);
      mergeToSubGraph(body.first, bodyInputRemove, constOp.get(), inputName);
      mergeToSubGraph(cond.first, condInputRemove, constOp.get(), inputName);
      removeInputs.insert(subI);

      modifiedSubGraph.insert(body.first);
      modifiedSubGraph.insert(cond.first);

      // Release no needed Const Memory
      constOp->main.Reset();
    }
    auto removeSubGraphInputs = [](ace::SubGraphProtoT* subGraph,
                                   const std::set<int>& inputRemove) {
      auto originInput = std::move(subGraph->inputs);
      subGraph->inputs.clear();
      for (auto index : originInput) {
        if (inputRemove.find(index) == inputRemove.end()) {
          subGraph->inputs.emplace_back(index);
        }
      }
    };
    removeSubGraphInputs(body.first, bodyInputRemove);
    removeSubGraphInputs(cond.first, condInputRemove);

    // Remove no use input for while op
    auto originIndexes = std::move(op->inputIndexes);
    auto aliInputs = std::move(param->aliases_inputs);
    for (int subI = 0; subI < originIndexes.size(); ++subI) {
      if (removeInputs.find(subI) == removeInputs.end()) {
        op->inputIndexes.emplace_back(originIndexes[subI]);
        param->aliases_inputs.emplace_back(std::move(aliInputs[subI]));
      }
    }
  }
  if (removeConstOpIndexes.empty()) {
    return false;
  }
  auto originOpLists = std::move(net->oplists);
  for (int i = 0; i < originOpLists.size(); ++i) {
    if (removeConstOpIndexes.find(i) == removeConstOpIndexes.end()) {
      net->oplists.emplace_back(std::move(originOpLists[i]));
    }
  }
  // Try Optimize Subgraph for more const op get
  auto* ctx = Global<OptimizeContext>::Get();
  std::unordered_map<std::string, VARP> empty;
  for (auto mutable_subgraph : modifiedSubGraph) {
    std::unique_ptr<ace::NetT> subnet(new ace::NetT);
    subnet->oplists = std::move(mutable_subgraph->nodes);
    subnet->tensorName = std::move(mutable_subgraph->tensors);
    subnet->sourceType = ctx->source;

    std::unique_ptr<ace::NetT> new_subnet = optimizeNetImpl(subnet, empty);
    mutable_subgraph->nodes = std::move(subnet->oplists);

    ace::SubGraphProtoT* new_subgraph = mutable_subgraph;
    for (int i = 0; i < mutable_subgraph->inputs.size(); ++i) {
      auto& name = subnet->tensorName[mutable_subgraph->inputs[i]];
      for (int v = 0; v < new_subnet->tensorName.size(); ++v) {
        if (new_subnet->tensorName[v] == name) {
          mutable_subgraph->inputs[i] = v;
          break;
        }
      }
    }
    for (int i = 0; i < mutable_subgraph->outputs.size(); ++i) {
      auto& name = subnet->tensorName[mutable_subgraph->outputs[i]];
      for (int v = 0; v < new_subnet->tensorName.size(); ++v) {
        if (new_subnet->tensorName[v] == name) {
          mutable_subgraph->outputs[i] = v;
          break;
        }
      }
    }
    mutable_subgraph->nodes = std::move(new_subnet->oplists);
    mutable_subgraph->tensors = std::move(new_subnet->tensorName);
  }
  return true;
}

}  // namespace Express
}  // namespace ace

using namespace ace;
using namespace ace::Express;

std::unique_ptr<ace::NetT> optimizeNet(std::unique_ptr<ace::NetT>& originNet,
                                       bool forTraining, modelConfig& config) {
  Global<modelConfig>::Reset(&config);
  //   if (originNet->sourceType == NetSource_TENSORFLOW) {
  //     GenerateSubGraph(originNet);
  //   }
  std::vector<ace::SubGraphProtoT*> subgraphs;
  for (auto& subgraph : originNet->subgraphs) {
    subgraphs.push_back(subgraph.get());
  }
  OptimizeContext ctx;
  ctx.subgraphs = subgraphs;
  ctx.is_training = forTraining;
  ctx.verbose = true;
  ctx.source = originNet->sourceType;
  ctx.completed_subgraphs = {};
  ctx.RunOptimize = optimizeNetImpl;

  Global<OptimizeContext>::Reset(&ctx);

  if (!originNet->subgraphs.empty()) {
    std::unordered_map<std::string, VARP> inputs;
    auto program = Program::create(originNet.get(), true);
    for (const auto& iter : program->vars()) {
      if (iter.first < originNet->tensorName.size() && iter.first >= 0) {
        inputs[originNet->tensorName[iter.first]] = iter.second;
      }
    }
    for (auto& subGraph : originNet->subgraphs) {
      CompleteSubGraph(inputs, subGraph.get());
    }
  }
  std::unordered_map<std::string, VARP> empty;
  std::unique_ptr<ace::NetT> net = ctx.RunOptimize(originNet, empty);
  fuseConstIntoSubgraph(net.get(), ctx.completed_subgraphs);
  for (auto* subgraph : ctx.completed_subgraphs) {
    net->subgraphs.emplace_back(subgraph);
  }
  return std::move(net);
}
