#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#include "glog/logging.h"
#include "tars/utils/flatbuffers.h"

namespace tars {

FlatbuffersModel::FlatbuffersModel(const std::string& path)
    : model_path_(path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    LOG(FATAL) << "Flatbuffers Model " << path << " not exists.";
  }

  // calc the length of the serialized tars model.
  file.seekg(0, std::ios::end);
  model_bytes_ = file.tellg();
  file.seekg(0, std::ios::beg);
  CHECK(model_bytes_ > 0) << "Flatbuffers Model " << path << " maybe empty.";

  char* buffer = new char[model_bytes_];
  file.read(buffer, model_bytes_);

  model_ = tars::UnPackNet(buffer);
  CHECK(model_->oplists.size() > 0);

  LOG(INFO) << "--------------------------------------";
  LOG(INFO) << "Model Path: " << model_path_;
  LOG(INFO) << "Model Bytes: " << model_bytes_;
  LOG(INFO) << "Model Ops: " << model_->oplists.size();
  for (int i = 0; i < model_->oplists.size(); ++i) {
    LOG(INFO) << "op[" << i << "] name: " << model_->oplists[i]->name
              << " type: " << model_->oplists[i]->type;
  }

  LOG(INFO) << "\n";
  delete[] buffer;

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = tars::Net::Pack(builder, model_.get());
  builder.Finish(offset);
  buffer_size_ = builder.GetSize();
  buffer_.reset(new uint8_t[buffer_size_], std::default_delete<uint8_t[]>());
  memcpy(buffer_.get(), builder.GetBufferPointer(), buffer_size_);

  model_.reset();
}

}  // namespace tars