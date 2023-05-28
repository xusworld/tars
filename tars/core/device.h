#ifndef TARS_CORE_DEVICE_H_
#define TARS_CORE_DEVICE_H_

#include <memory>

#include "ir/current/model_generated.h"
#include "tars/core/macro.h"
#include "tars/core/operation.h"
#include "tars/core/status.h"
#include "tars/core/tensor.h"

namespace tars {

enum class DeviceType {
  CPU = 0,  // Linux x86 device
  GPU = 1,  // Nvidia GPU device
  NPU = 2   // other device
};

class Device {
 public:
  Device();

  virtual ~Device();

  // sync everything
  virtual Status sync() = 0;

  // dynamic shape
  virtual Status dynamic() = 0;

  // acquire buffer from memory pool
  virtual Status acquire() = 0;

  // release buffer to memory pool
  virtual Status release() = 0;

  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op) = 0;

 private:
  // the id of machine
  int32_t machine_id_;
  // the id of device
  int32_t device_id_;
  // the type of device
  DeviceType dtype_;
  // the runtime of device
  std::shared_ptr<Runtime> runtime_;
};

}  // namespace tars

#endif  // TARS_CORE_DEVICE_H_