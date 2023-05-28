#pragma once

#include "tars/core/macro.h"
#include "tars/core/runtime.h"

namespace tars {

struct WorkspaceContext {};

struct OpContext {
  // op
  const Op* op;
  // input tensors
  std::vector<Tensor*> inputs;
  // output tensors
  std::vector<Tensor*> outputs;
};

class Workspace final {
 public:
  DISABLE_COPY_MOVE_ASSIGN(Workspace);

  Workspace(const WorkspaceContext& ctx);

  ~Workspace();

  // build a workspace from deserialized output (name, device) pairs
  Status build();

  // invoke
  Status invoke();

  // allocate necessary memory
  Status alloc();

 private:
  Runtime* runtime_;
};

}  // namespace tars