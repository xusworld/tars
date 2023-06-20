#include <glog/logging.h>

#include <fstream>
#include <memory>

#include "ace/schema/ace_generated.h"
#include "ace/schema/extra_info_generated.h"
#include "config_file.h"
#include "converter.h"

template <typename T>
static void storeWeight(std::ofstream& fs, std::vector<T>& weight,
                        std::vector<int64_t>& external, int64_t& offset) {
  if (external.empty()) {
    external.push_back(offset);
  }
  int64_t size = weight.size() * sizeof(T);
  fs.write(reinterpret_cast<const char*>(weight.data()), size);
  weight.clear();
  external.push_back(size);
  offset += size;
}

static void RemoveAndStoreParam(std::unique_ptr<ace::OpT>& op,
                                std::ofstream& fs, int64_t& offset) {
  const auto opType = op->type;
  switch (opType) {
    case ace::OpType_Convolution:
    case ace::OpType_Deconvolution:
    case ace::OpType_ConvolutionDepthwise: {
      auto param = op->main.AsConvolution2D();
      storeWeight<float>(fs, param->weight, param->external, offset);
      storeWeight<float>(fs, param->bias, param->external, offset);
      break;
    }
    case ace::OpType_Scale: {
      auto param = op->main.AsScale();
      storeWeight<float>(fs, param->scaleData, param->external, offset);
      if (!param->biasData.empty()) {
        storeWeight<float>(fs, param->biasData, param->external, offset);
      }
      break;
    }
    case ace::OpType_LayerNorm: {
      auto param = op->main.AsLayerNorm();
      if (!param->gamma.empty() && !param->beta.empty()) {
        storeWeight<float>(fs, param->gamma, param->external, offset);
        storeWeight<float>(fs, param->beta, param->external, offset);
      }
      break;
    }
    case ace::OpType_TrainableParam:
    case ace::OpType_Const: {
      auto param = op->main.AsBlob();
      switch (param->dataType) {
        case ace::DataType_DT_FLOAT:
          storeWeight<float>(fs, param->float32s, param->external, offset);
          break;
        case ace::DataType_DT_INT32:
          storeWeight<int>(fs, param->int32s, param->external, offset);
          break;
        case ace::DataType_DT_UINT8:
          storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
          break;
        case ace::DataType_DT_INT8:
          storeWeight<int8_t>(fs, param->int8s, param->external, offset);
          break;
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
}

bool saveExternalData(std::unique_ptr<ace::NetT>& netT,
                      const std::string& extraFileName) {
  std::ofstream extraFile(extraFileName, std::ios::binary);
  if (!extraFile.is_open()) {
    return false;
  }
  int64_t offset = 0;
  for (auto& op : netT->oplists) {
    RemoveAndStoreParam(op, extraFile, offset);
  }
  for (auto& subgraph : netT->subgraphs) {
    for (auto& op : subgraph->nodes) {
      RemoveAndStoreParam(op, extraFile, offset);
    }
  }
  extraFile.close();
  return true;
}
/*
void genStaticModel(CommandBuffer buffer, const std::string& modelName,
                    std::map<Tensor*, std::pair<std::string, int>>& tensorNames,
                    std::vector<std::string>&& outputNames,
                    const Net* originNetInfo) {
  MNN_PRINT("gen Static Model ... \n");
  std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
  netT->outputName = std::move(outputNames);
  netT->usage = Usage_INFERENCE_STATIC;
  std::map<Tensor*, int> tensorMap;
  // Add tensorName to new netT
  netT->tensorName.resize(tensorNames.size());
  std::vector<std::unique_ptr<OpT>> inputOps;
  for (auto& iter : tensorNames) {
    netT->tensorName[iter.second.second] = iter.second.first;
    tensorMap.insert(std::make_pair(iter.first, iter.second.second));
    if (TensorUtils::getDescribe(iter.first)->usage ==
        MNN::Tensor::InsideDescribe::INPUT) {
      std::unique_ptr<OpT> input(new OpT);
      input->type = OpType_Input;
      input->name = iter.second.first;
      input->outputIndexes = {iter.second.second};
      input->main.value = new InputT;
      input->main.type = OpParameter_Input;
      input->main.AsInput()->dims = iter.first->shape();
      input->main.AsInput()->dformat =
          TensorUtils::getDescribe(iter.first)->dimensionFormat;
      auto type = iter.first->getType();
      if (type.code == halide_type_float) {
        if (type.bits == 32) {
          input->main.AsInput()->dtype = DataType_DT_FLOAT;
        } else if (type.bits == 16) {
          input->main.AsInput()->dtype = DataType_DT_HALF;
        }
      } else if (type.code == halide_type_int) {
        if (type.bits == 32) {
          input->main.AsInput()->dtype = DataType_DT_INT32;
        } else if (type.bits == 16) {
          input->main.AsInput()->dtype = DataType_DT_INT16;
        } else if (type.bits == 8) {
          input->main.AsInput()->dtype = DataType_DT_INT8;
        }
      } else if (type.code == halide_type_uint) {
        if (type.bits == 16) {
          input->main.AsInput()->dtype = DataType_DT_UINT16;
        } else if (type.bits == 8) {
          input->main.AsInput()->dtype = DataType_DT_UINT8;
        }
      }
      inputOps.emplace_back(std::move(input));
    }
  }
  // add Tensors to netT
  for (auto& iterP : buffer.command) {
    auto& iter = *iterP;
    std::function<void(Tensor*)> insertTensor = [&](Tensor* t) {
      if (tensorMap.find(t) == tensorMap.end()) {
        int index = static_cast<int>(tensorMap.size());
        tensorMap.insert(std::make_pair(t, index));
        std::string tensorName = "ExtraTensor_" + std::to_string(index);
        netT->tensorName.push_back(tensorName);
      }
    };
    for (auto& t : iter.inputs) {
      insertTensor(t);
    }
    for (auto& t : iter.outputs) {
      insertTensor(t);
    }
  }
  // add tensors' describe to netT
  for (auto tensorPair : tensorMap) {
    auto tensor = tensorPair.first;
    auto index = tensorPair.second;
    // FUNC_PRINT(index);
    auto des = TensorUtils::getDescribe(tensor);
    if (des->usage == Tensor::InsideDescribe::CONSTANT) {
      std::unique_ptr<OpT> op(new OpT);
      op->type = OpType_Const;
      auto blob = new BlobT;
      op->main.type = OpParameter_Blob;
      op->main.value = blob;
      blob->dataFormat = des->dimensionFormat;
      for (int d = 0; d < tensor->dimensions(); d++) {
        blob->dims.push_back(tensor->buffer().dim[d].extent);
      }
      if (tensor->getType() == halide_type_of<float>()) {
        blob->dataType = DataType_DT_FLOAT;
        for (int i = 0; i < tensor->elementSize(); i++) {
          blob->float32s.push_back(tensor->host<float>()[i]);
        }
      } else {
        CONSTANT_COPY(INT8, int8);
        CONSTANT_COPY(UINT8, uint8);
        CONSTANT_COPY(INT32, int32)
        CONSTANT_COPY(INT64, int64);
      }
      op->outputIndexes.push_back(index);
      netT->oplists.emplace_back(std::move(op));
    }
    auto describe =
        std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
    describe->index = index;
    describe->blob = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
    auto& blob = describe->blob;
    blob->dataFormat = des->dimensionFormat;
    if (tensor->getType() == halide_type_of<float>()) {
      blob->dataType = DataType_DT_FLOAT;
    } else {
      SET_TYPE(INT8, int8)
    }
    SET_TYPE(UINT8, uint8)
  }
  SET_TYPE(INT32, int32)
}
SET_TYPE(INT64, int64)

for (int d = 0; d < tensor->dimensions(); d++) {
  describe->blob->dims.push_back(tensor->buffer().dim[d].extent);
}
auto tensorDes = TensorUtils::getDescribe(tensor);
if (nullptr != tensorDes->quantAttr) {
  describe->quantInfo.reset(new TensorQuantInfoT);
  describe->quantInfo->max = tensorDes->quantAttr->max;
  describe->quantInfo->min = tensorDes->quantAttr->min;
  describe->quantInfo->zero = tensorDes->quantAttr->zero;
  describe->quantInfo->scale = tensorDes->quantAttr->scale;
}
for (auto& reg : des->regions) {
  auto regionT = std::unique_ptr<MNN::RegionT>(new MNN::RegionT);
  regionT->src = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
  regionT->dst = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
  regionT->src->offset = reg.src.offset;
  regionT->dst->offset = reg.dst.offset;
  for (int s = 0; s < 3; s++) {
    regionT->src->stride.push_back(reg.src.stride[s]);
    regionT->dst->stride.push_back(reg.dst.stride[s]);
    regionT->size.push_back(reg.size[s]);
  }
  describe->regions.emplace_back(std::move(regionT));
}
netT->extraTensorDescribe.emplace_back(std::move(describe));

// add op to netT
for (auto&& iter : inputOps) {
  netT->oplists.emplace_back(std::move(iter));
}
int idx = 0;
for (auto& iterP : buffer.command) {
  auto& iter = *iterP;
  auto opt = iter.op->UnPack();
  if (opt->name.size() <= 0) {
    opt->name = std::string("Geometry_") + MNN::EnumNameOpType(opt->type) +
                std::to_string(idx++);
  }
  opt->inputIndexes.resize(iter.inputs.size());
  opt->outputIndexes.resize(iter.outputs.size());
  for (int i = 0; i < iter.outputs.size(); i++) {
    opt->outputIndexes[i] = tensorMap[iter.outputs[i]];
  }
  for (int i = 0; i < iter.inputs.size(); i++) {
    opt->inputIndexes[i] = tensorMap[iter.inputs[i]];
  }
  netT->oplists.emplace_back(std::move(opt));
}
_RemoveUnusefulNodes(netT);
netT->usage = Usage_INFERENCE_STATIC;
netT->sourceType = originNetInfo->sourceType();
if (nullptr != originNetInfo->bizCode()) {
  netT->bizCode = originNetInfo->bizCode()->str();
}
if (nullptr != originNetInfo->mnn_uuid()) {
  netT->mnn_uuid = originNetInfo->mnn_uuid()->str();
}
netT->extraInfo.reset(new ExtraInfoT);
netT->extraInfo->version = MNN_VERSION;
// write netT to file
flatbuffers::FlatBufferBuilder builderOutput(1024);
auto len = MNN::Net::Pack(builderOutput, netT.get());
builderOutput.Finish(len);
int sizeOutput = builderOutput.GetSize();
auto bufferOutput = builderOutput.GetBufferPointer();
std::ofstream output(modelName, std::ofstream::binary);
output.write((const char*)bufferOutput, sizeOutput);
}

void converToStaticModel(const ace::Net* net,
                         std::map<std::string, std::vector<int>>& inputConfig,
                         std::string aceFile) {
  // set a backend and context to run resize
  ScheduleConfig config;
  config.type = ace_FORWARD_CPU;
  BackendConfig backendConfig;
  backendConfig.precision = BackendConfig::Precision_High;
  config.backendConfig = &backendConfig;
  Backend::Info compute;
  compute.type = config.type;
  compute.numThread = config.numThread;
  compute.user = config.backendConfig;
  const RuntimeCreator* runtimeCreator(aceGetExtraRuntimeCreator(compute.type));
  std::unique_ptr<Runtime> runtime(runtimeCreator->onCreate(compute));
  std::shared_ptr<Backend> backend(runtime->onCreate());
  BackendConfig defaultConfig;
  defaultConfig.flags = 4;
  std::shared_ptr<Backend> defaultBackend(runtime->onCreate(&defaultConfig));
  std::vector<std::shared_ptr<Tensor>> allTensors;
  allTensors.resize(net->tensorName()->size());
  ErrorCode code = NO_ERROR;
  initConstTensors(allTensors, net, defaultBackend.get(), code);
  if (NO_ERROR != code) {
    ace_ERROR("Init tensor error code = %d\n", code);
    return;
  }
  bool valid = initTensors(allTensors, net);
  // set tensors' shape by inputConfig
  for (int i = 0; i < allTensors.size(); i++) {
    auto name = net->tensorName()->GetAsString(i)->str();
    if (inputConfig.find(name) != inputConfig.end()) {
      auto& dims = inputConfig[name];
      allTensors[i]->buffer().dimensions = dims.size();
      for (int j = 0; j < dims.size(); j++) {
        allTensors[i]->setLength(j, dims[j]);
      }
    }
  }
  std::vector<Schedule::OpCacheInfo> infos;
  initPipelineInfosFromNet(infos, net, allTensors);
  GeometryComputer::Context ctx(defaultBackend);
  // resize the session's info and store to buffer
  std::vector<Tensor*> constTensors;
  GeometryComputerUtils::buildConstantTensors(infos);
  GeometryComputerUtils::shapeComputeAndGeometryTransform(
      infos, ctx, defaultBackend, runtime->onGetCompilerType());
  std::map<Tensor*, std::pair<std::string, int>> tensorName;
  for (int i = 0; i < net->tensorName()->size(); i++) {
    tensorName[allTensors[i].get()] =
        std::make_pair(net->tensorName()->GetAsString(i)->str(), i);
  }
  std::vector<std::string> outputNames;
  if (net->outputName() != nullptr) {
    for (int i = 0; i < net->outputName()->size(); ++i) {
      outputNames.emplace_back(net->outputName()->GetAsString(i)->str());
    }
  } else {
    for (int i = 0; i < net->tensorName()->size(); i++) {
      if (TensorUtils::getDescribe(allTensors[i].get())->usage ==
          ace::Tensor::InsideDescribe::OUTPUT) {
        outputNames.emplace_back(net->tensorName()->GetAsString(i)->str());
      }
    }
  }
  CommandBuffer newBuffer;
  for (auto& info : infos) {
    auto& buf = info.executeBuffer;
    newBuffer.command.insert(newBuffer.command.end(), buf.command.begin(),
                             buf.command.end());
    newBuffer.extras.insert(newBuffer.extras.end(), buf.extras.begin(),
                            buf.extras.end());
  }
  // store buffer to STATIC model file
  genStaticModel(newBuffer, aceFile, tensorName, std::move(outputNames), net);
}
*/

int writeFb(std::unique_ptr<ace::NetT>& netT, const std::string& aceModelFile) {
  // std::string compressFileName = config.compressionParamsFile;
  // Compression::Pipeline proto;
  // if (compressFileName != "") {
  //   std::fstream input(compressFileName.c_str(),
  //                      std::ios::in | std::ios::binary);
  //   if (!proto.ParseFromIstream(&input)) {
  //     ace_ERROR("Failed to parse compression pipeline proto.\n");
  //   }
  // }

  // addUUID(netT, proto);

  // add version info to model
  netT->extraInfo.reset(new ace::ExtraInfoT);
  netT->extraInfo->version = "0.1.1";
  // if (!config.authCode.empty()) {
  //   // add auth code to model
  //   netT->extraInfo->name = config.authCode;
  // }
  // if (config.detectSparseSpeedUp) {
  //   addSparseInfo(netT, proto);
  // }
  // if (config.compressionParamsFile != "") {
  //   fullQuantAndCoding(netT, proto);
  // }

  // weightQuantAndCoding(netT, config);

  std::set<std::string> notSupportOps;
  auto CheckIfNotSupported = [&](const std::unique_ptr<ace::OpT>& op) {
    if (op->type == ace::OpType_Extra) {
      LOG(INFO) << "op->main.AsExtra()->engine: " << op->main.AsExtra()->engine;
      if (op->main.AsExtra()->engine != "ace") {
        notSupportOps.insert(op->main.AsExtra()->engine +
                             "::" + op->main.AsExtra()->type);
      }
    }
  };
  LOG(INFO) << "Main Graph oplist size: " << netT->oplists.size();
  for (auto& op : netT->oplists) {
    CheckIfNotSupported(op);
  }
  LOG(INFO) << "Sub Graph Size: " << netT->subgraphs.size();
  for (auto& subgraph : netT->subgraphs) {
    for (auto& op : subgraph->nodes) {
      CheckIfNotSupported(op);
    }
  }

  std::ostringstream notSupportInfo;
  if (!notSupportOps.empty()) {
    for (auto name : notSupportOps) {
      notSupportInfo << name << " | ";
    }
    auto opNames = notSupportInfo.str();
    LOG(FATAL) << "These Op Not Support: "
               << opNames.substr(0, opNames.size() - 2);
    return 1;
  }

  // dump input and output tensor name
  {
    std::set<int> inputIdx, outputIdx, realInput, realOutput;
    for (const auto& op : netT->oplists) {
      for (auto i : op->inputIndexes) {
        inputIdx.insert(i);
      }
      for (auto o : op->outputIndexes) {
        outputIdx.insert(o);
        if (op->type == ace::OpType_Input) {
          realInput.insert(o);
        }
      }
    }
    std::set_difference(outputIdx.begin(), outputIdx.end(), inputIdx.begin(),
                        inputIdx.end(),
                        std::inserter(realOutput, realOutput.begin()));
    std::cout << "inputTensors : [ ";
    for (int i : realInput) {
      std::cout << netT->tensorName[i] << ", ";
    }
    std::cout << "]\noutputTensors: [ ";
    if (netT->outputName.size() > 0) {
      for (auto& o : netT->outputName) {
        std::cout << o << ", ";
      }
    } else {
      for (int i : realOutput) {
        std::cout << netT->tensorName[i] << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  flatbuffers::FlatBufferBuilder builderOutput(1024);
  builderOutput.ForceDefaults(true);
  // if (config.saveExternalData) {
  //   bool res = saveExternalData(netT, aceModelFile + ".weight");
  //   if (!res) {
  //     LOG(FATAL) << "Write Weight to External Data Failed.";
  //   }
  // }
  auto len = ace::Net::Pack(builderOutput, netT.get());
  builderOutput.Finish(len);
  int sizeOutput = builderOutput.GetSize();
  auto bufferOutput = builderOutput.GetBufferPointer();

  // if (config.saveStaticModel && netT->usage != ace::Usage_INFERENCE_STATIC) {
  //   std::map<std::string, std::vector<int>> inputConfig;
  //   // get config to set input size
  //   if (config.inputConfigFile.size() > 0) {
  //     ConfigFile conf(config.inputConfigFile);
  //     auto numOfInputs = conf.Read<int>("input_size");
  //     auto inputNames =
  //         splitNames(numOfInputs, conf.Read<std::string>("input_names"));
  //     auto inputDims =
  //         splitDims(numOfInputs, conf.Read<std::string>("input_dims"));
  //     for (int i = 0; i < numOfInputs; i++) {
  //       inputConfig.insert(std::make_pair(inputNames[i], inputDims[i]));
  //     }
  //   }
  //   const ace::Net* net = flatbuffers::GetRoot<ace::Net>(bufferOutput);
  //   // converToStaticModel(net, inputConfig, aceModelFile);
  // } else {
  std::ofstream output(aceModelFile, std::ofstream::binary);
  output.write((const char*)bufferOutput, sizeOutput);

  if (!netT->subgraphs.empty()) {
    LOG(INFO) << "The model has subgraphs, please use ace::Module to run it";
  }

  return 0;
}

int main() {
  LOG(INFO) << "Ace model parser, only support onnx model to ace model."
            << std::endl;
  const std::string model = "/data/lukedong/Ace/models/mobilenetv2-7.onnx";

  std::unique_ptr<ace::NetT> netT = std::unique_ptr<ace::NetT>(new ace::NetT());
  ace::converter::Onnx2AceNet(model, "1", netT);

  writeFb(netT, "/data/lukedong/Ace/models/mobilenetv2-7.ace");
  return 0;
}