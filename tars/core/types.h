#pragma once

#include <ostream>
#include <string>

namespace tars {

/** backend buffer storage type */
enum StorageType {
  /**
   use NOT reusable memory.
   - allocates memory when `onAcquireBuffer` is called.
   - releases memory when `onReleaseBuffer` is called or when the backend is
   deleted.
   - do NOTHING when `onClearBuffer` is called.
   */
  STATIC,
  /**
   use reusable memory.
   - allocates or reuses memory when `onAcquireBuffer` is called. prefers
   reusing.
   - collects memory for reuse when `onReleaseBuffer` is called.
   - releases memory when `onClearBuffer` is called or when the backend is
   deleted.
   */
  DYNAMIC,
  /**
   use NOT reusable memory.
   - allocates memory when `onAcquireBuffer` is called.
   - do NOTHING when `onReleaseBuffer` is called.
   - releases memory when `onClearBuffer` is called or when the backend is
   deleted.
   */
  DYNAMIC_SEPERATE
};

enum class TensorKind {
  None,
  Input,   // input tensor
  Output,  // output tensor
  Const,   // constant tensor
};

enum class MemcpyKind {
  H2D,  // host to device
  D2H,  // device to host
  D2D,  // device to device
  H2H,  // host to host
};

enum DataFormat : int8_t {
  DataFormat_NONE = 0,
  DataFormat_NCHW = 1,
  DataFormat_NHWC = 2,
  DataFormat_NC4HW4 = 3,
  DataFormat_NHWC4 = 4,
  DataFormat_MIN = DataFormat_NONE,
  DataFormat_MAX = DataFormat_NHWC4
};

enum class RuntimeType { CPU = 0, CUDA = 1 };

// inline std::ostream& operator<<(std::ostream& os, RuntimeType& type) {
//   if (type == RuntimeType::CPU) {
//     os << "CPU";
//   }

//   return os;
// }

// returns the memcpy kind string
std::string MemcpyKindToString(const MemcpyKind kind);

// returns the runtime type string
std::string RuntimeTypeToString(const RuntimeType type);
}  // namespace tars