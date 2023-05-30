#pragma once

#include "tars/core/device.h"
#include "tars/core/memory/memory_pool.h"

namespace tars {
namespace device {

class CpuDevice : public Device {
 public:
  CpuDevice();

  virtual ~CpuDevice();

  // sync everything
  virtual Status sync() = 0;

  // dynamic shape
  virtual Status dynamic();

  // acquire buffer from memory pool
  virtual Status acquire();

  // release buffer to memory pool
  virtual Status release();

  virtual Operation* onCreate(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs,
                              const tars::Op* op);

  // returns machine id which the device belongs to
  int32_t machine_id() const { return 0; }

  // returns device id
  int32_t device_id() const { return 0; }

  // returns device type
  DeviceType device_type() const { return dtype_; }

 private:
  // the id of machine
  int32_t machine_id_ = 0;
  // the id of device
  int32_t device_id_ = 0;
  // the type of device
  DeviceType dtype_ = DeviceType::CPU;
  // the runtime of device
  std::shared_ptr<Runtime> runtime_;
  // the memory pool of device
  MemoryPool<RuntimeType::CPU> mem_pool_;
};

}  // namespace device
}  // namespace tars