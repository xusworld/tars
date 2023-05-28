#ifndef TARS_CORE_OPERATION_H_
#define TARS_CORE_OPERATION_H_

#include "tars/core/macro.h"
#include "tars/core/status.h"

namespace tars {

class Device;

class Operation {
 public:
  DISABLE_COPY_MOVE_ASSIGN(Operation);

  virtual ~Operation() = default;

  virtual Status run() = 0;

  virtual Device* device() const { return device_; }

 private:
  Device* device_;
};

}  // namespace tars

#endif  // TARS_CORE_OPERATION_H_