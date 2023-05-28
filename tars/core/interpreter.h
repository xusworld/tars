#ifndef TARS_CORE_INTERPRETER_H_
#define TARS_CORE_INTERPRETER_H_

#include <memory>
#include <mutex>
#include <vector>

#include "glog/logging.h"
#include "ir/current/model_generated.h"
#include "tars/core/macro.h"
#include "tars/core/session.h"
#include "tars/core/status.h"
#include "tars/utils/flatbuffers.h"

namespace tars {

// model configs
class ModelConf {};

// Tars model interpreter.
class Interpreter final {
 public:
  DISABLE_COPY_MOVE_ASSIGN(Interpreter);

  Interpreter(const FlatbuffersModel& model);

  ~Interpreter() {
    delete graph_;
    delete buffer_;
  }

  Status makeSession() {
    CHECK(buffer_ != nullptr) << "a empty model pointer";
    CHECK(buffer_size_ > 0) << "a empty model";
    return Status::OK();
  }

  Status initInputs();

  Status initOutputs() { return Status::OK(); }

  Status build_workspace();

  Status initOps() { return Status::OK(); }

 private:
  // mutex
  std::mutex mutex_;
  // dynamic shape or static shape
  bool is_dynamic_shape_;
  // flatbuffers model
  const Net* graph_ = nullptr;
  // model buffer
  uint8_t* buffer_;
  int32_t buffer_size_;

  // session list
  std::vector<Session> sessions_;
  // all tensors in this model, use vector index as tensor id
  std::vector<Tensor*> tensors_;
};

}  // namespace tars

#endif  // TARS_CORE_INTERPRETER_H_