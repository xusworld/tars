#ifndef TARS_UTILS_FLATBUFFERS_H_
#define TARS_UTILS_FLATBUFFERS_H_

#include <memory>
#include <string>

#include "ir/current/model_generated.h"
#include "tars/core/macro.h"

namespace tars {
namespace {

// the pointer of serialized tars model
using ModelPtr = std::unique_ptr<tars::NetT>;

}  // namespace

class FlatbuffersModel final {
 public:
  DISABLE_COPY_MOVE_ASSIGN(FlatbuffersModel);

  FlatbuffersModel(const std::string& path);

  ~FlatbuffersModel() = default;

  // returns flatbuffers model's buffer pointer
  void* buffer() const { return buffer_.get(); }

  // returns flatbuffers model's buffer pointer
  int32_t buffer_size() const { return buffer_size_; }

 private:
  std::string model_path_;
  int32_t model_bytes_;
  std::unique_ptr<tars::NetT> model_;
  std::shared_ptr<uint8_t> buffer_;
  int32_t buffer_size_;
};

}  // namespace tars

#endif  // TARS_UTILS_FLATBUFFERS_H_