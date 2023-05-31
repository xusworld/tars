#include <memory>
#include <unordered_map>

#include "glog/logging.h"
#include "tars/core/interpreter.h"

namespace tars {

Interpreter::Interpreter(const FlatbuffersModel& fbs_model) {
  CHECK(fbs_model.buffer() != nullptr) << "a empty model pointer";
  CHECK(fbs_model.buffer_size() > 0) << "a empty model";

  buffer_ = reinterpret_cast<uint8_t*>(fbs_model.buffer());
  buffer_size_ = fbs_model.buffer_size();

  flatbuffers::Verifier verify(buffer_, buffer_size_);
  if (!VerifyNetBuffer(verify)) {
    LOG(FATAL) << "Tars model buffer error.";
  }

  graph_ = GetNet(buffer_);
  LOG(INFO) << "dynamic shape flag:  " << graph_->usage();
}

Status Interpreter::initInputs() {
  std::vector<Tensor*> tensors;
  tensors.resize(graph_->tensorName()->size());

  LOG(INFO) << "net->tensorName()->size(): " << graph_->tensorName()->size();
  LOG(INFO) << "net->oplists()->size(): " << graph_->oplists()->size();
  // handle constant op in graph
  for (int i = 0; i < graph_->oplists()->size(); ++i) {
    auto op = graph_->oplists()->GetAs<Op>(i);

    if (op->type() != OpType_Const) {
      // LOG(INFO) << "op " << op->name()->str() << " not constant, skip...";
      continue;
    }
    // LOG(INFO) << "op " << op->name()->str() << " is constant, process...";
    auto index = op->outputIndexes()->data()[0];
    // LOG(INFO) << "index: " << index;
    CHECK(i < tensors.size())
        << " op index: " << i
        << " is greater than tensors size: " << tensors.size();
    auto parameter = op->main_as_Blob();
    auto output = tensors[i];
    bool zeroShape = false;

    // get shape info from blob
    if (parameter->dims() != nullptr) {
      auto size = parameter->dims()->size();
      LOG(INFO) << "size: " << size;
    }
  }

  // handle tensor in graph
  for (int i = 0; i < graph_->oplists()->size(); i++) {
    auto op = graph_->oplists()->GetAs<Op>(i);
    if (op->type() == OpType_Input) {
      LOG(INFO) << "Model Input Name: " << op->name()->str();
      CHECK(op->outputIndexes() != nullptr) << "input op's output is nullptr.";
      CHECK(op->outputIndexes()->size() == 1)
          << "input op's output is not single.";
      auto index = op->outputIndexes()->data()[0];
      LOG(INFO) << "index: " << index;

      auto tensor = Tensor();
      auto option = op->main_as_Input();
      auto dims = option->dims();
      LOG(INFO) << "dims: " << dims;

      TensorShape shape;
      for (int i = 0; i < dims->size(); ++i) {
        int dim = dims->data()[i];
        LOG(INFO) << "initTensors| dim[" << i << "] = " << dim;
        if (dim <= 0) dim = 1;
        shape[i] = dim;
      }

      LOG(INFO) << "tensor reshape to " << shape;
      tensor.reshape(shape);
      tensor.astype(option->dtype());
      tensor.set_dformat(DataFormat_NCHW);

      tensors.push_back(std::move(&tensor));
      LOG(INFO) << "done";
    }
  }
  LOG(INFO) << "handle describes";

  // 删除 tensor descibe
  auto describes = graph_->extraTensorDescribe();
  LOG(INFO) << "describes: " << describes;
  if (describes != nullptr) {
    LOG(INFO) << "describes->size: " << describes->size();
  }

  LOG(INFO) << "Interpreter::initInputs done";

  return Status::OK();
}

Status Interpreter::build_workspace() {
  LOG(INFO) << "Build workspace";
  const auto ops_size = graph_->oplists()->size();

  std::vector<Op*> ops;
  ops.clear();
  ops.reserve(ops_size);

  for (int i = 0; i < ops_size; i++) {
    auto op = graph_->oplists()->GetAs<Op>(i);
    // ops.emplace_back(op);
  }

  for (auto op : ops) {
    // build context for operator
    OpContext ctx;
    // pointer
    ctx.op = op;
    // inputs
    if (op->inputIndexes() != nullptr) {
      auto indices = op->inputIndexes()->data();
      auto size = op->inputIndexes()->size();
      for (int i = 0; i < size; i++) {
        ctx.inputs.push_back(tensors_[indices[i]]);
      }
    }
    // outputs
    if (op->outputIndexes() != nullptr) {
      auto indices = op->outputIndexes()->data();
      auto size = op->outputIndexes()->size();
      for (int i = 0; i < size; i++) {
        ctx.outputs.push_back(tensors_[indices[i]]);
      }
    }
    // shape inference
  }

  std::set<int> input_indices;
  std::set<int> output_indices;

  for (auto op : ops) {
    // build context for operator
    OpContext ctx;
    // pointer
    ctx.op = op;
    // inputs
    if (op->inputIndexes() != nullptr) {
      auto indices = op->inputIndexes()->data();
      auto size = op->inputIndexes()->size();
      for (int i = 0; i < size; i++) {
        input_indices.insert(indices[i]);
      }
    }
    // outputs
    if (op->outputIndexes() != nullptr) {
      auto indices = op->outputIndexes()->data();
      auto size = op->outputIndexes()->size();
      for (int i = 0; i < size; i++) {
        output_indices.insert(indices[i]);
      }
    }
  }

  // 2. the index in outputIndexes/inputIndexed but not in
  // inputIndexes/outputIndexes is output/input
  std::set<int> inputs;
  std::set<int> outputs;
  std::set_difference(output_indices.begin(), output_indices.end(),
                      input_indices.begin(), input_indices.end(),
                      std::inserter(outputs, outputs.begin()));
  std::set_difference(input_indices.begin(), input_indices.end(),
                      output_indices.begin(), output_indices.end(),
                      std::inserter(inputs, inputs.begin()));

  for (const auto input : inputs) {
    tensors_[input]->set_kind(TensorKind::Input);
  }

  for (const auto output : outputs) {
    tensors_[output]->set_kind(TensorKind::Output);
  }

  // add output index by config info and outputName
  std::unordered_map<std::string, int> name2Idx;
  for (int i = 0; i < graph_->tensorName()->size(); ++i) {
    name2Idx[graph_->tensorName()->Get(i)->str()] = i;
  }

  return Status::OK();
}

}  // namespace tars