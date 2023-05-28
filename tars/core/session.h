#pragma once

#include <map>

#include "tars/core/device.h"
#include "tars/core/macro.h"
#include "tars/core/runtime.h"
#include "tars/core/status.h"
#include "tars/core/tensor.h"
#include "tars/core/workspace.h"

namespace tars {

class Session final {
 public:
  DISABLE_COPY_MOVE_ASSIGN(Session);

  Session() = default;

  ~Session() = default;

  Status run();

  Status dynamic();

  // returns device of the session
  // Device* device() const { return device; }

  // returns runtime of the session
  Runtime* runtime() const { return runtime_; }

  // return workspace of the session
  Workspace* workspace() const { return workspace_; }

  // return inputs of the session
  std::vector<Tensor*> inputs() const { return inputs_; }

  // return outputs of the session
  std::vector<Tensor*> outputs() const { return outputs_; }

 private:
  Device* device_;
  Runtime* runtime_;
  Workspace* workspace_;

  std::vector<Tensor*> inputs_;
  std::vector<Tensor*> outputs_;
  std::map<std::string, Tensor*> name2inputs_;
  std::map<std::string, Tensor*> name2outputs_;
};

}  // namespace tars