#pragma once

#include <memory>
#include <vector>

#include "../memory/allocator.h"
#include "../memory/buffer.h"
#include "../memory/memory_manager.h"
#include "../status.h"

namespace ace {

constexpr const char *kNamespaceTensor = "tensor";
constexpr const char *kNamespaceScratch = "scratch";

// Runtime interface, a pure virtual class.
class Runtime {
 public:
  explicit Runtime();
  virtual ~Runtime();

  // Memory

  // Thread

  // TODO(luxuhui): should be changed to RuntimeConfig
  virtual Status Init(const MaceEngineCfgImpl *engine_config,
                      const MemoryType mem_type);

  virtual Status BeforeRun(MaceEngineCfgImpl *config);
  virtual Status AfterRun();

  virtual Status MapBuffer(Buffer *buffer, bool wait_for_finish);
  virtual Status UnMapBuffer(Buffer *buffer);
  virtual bool CanReuseBuffer(const Buffer *buffer,
                              const std::vector<index_t> &shape,
                              const BufferContentType content_type,
                              const unsigned int content_param);

  virtual RuntimeType GetRuntimeType() = 0;
  virtual RuntimeSubType GetRuntimeSubType();
  virtual MemoryType GetBaseMemoryType();
  virtual MemoryType GetUsedMemoryType();

  utils::ThreadPool &thread_pool();

  Status AllocateBufferForTensor(Tensor *tensor, BufRentType rent_type,
                                 Buffer *slice_parent = nullptr,
                                 index_t offset = 0);
  void ReleaseBufferForTensor(Tensor *tensor, const BufRentType rent_type);

  std::unique_ptr<Buffer> ObtainBuffer(const MemInfo &info,
                                       BufRentType rent_type);
  void ReleaseBuffer(Buffer *buffer, BufRentType rent_type);
  void ReleaseAllBuffer(BufRentType rent_type, bool del_buf = false);

  virtual std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) = 0;

  virtual std::vector<index_t> ComputeBufDimFromTensorDim(
      const std::vector<index_t> &dims, const MemoryType mem_type,
      const BufferContentType content_type, const unsigned int content_param);
  virtual DataType GetComputeDataType(const NetDef &net_def,
                                      const ConstTensor &const_tensor);

  virtual MemoryManager *GetMemoryManager(const MemoryType mem_type) = 0;

  void SetBufferToTensor(std::unique_ptr<Buffer> buffer, Tensor *tensor);

  // for inter buffers' release and re-allocate
  void ReleaseIntermediateBuffer(const BaseEngine *engine);
  void OnAllocateIntermediateBuffer(const BaseEngine *engine);
  void OnIntermediateBufferUsed(const BaseEngine *engine);
  bool IntermediateBufferCreated(const BaseEngine *engine) const;
  bool IntermediateBufferStable(const OpContext *op_context) const;

 protected:
  utils::ThreadPool *thread_pool_;

  // for inter buffers' release and re-allocate
  std::unordered_map<const void *, int> inter_mem_state_map_;
  bool has_ever_released_inter_mem_;  // For acceleration
};

}  // namespace ace

#endif  // MACE_CORE_RUNTIME_RUNTIME_H_
