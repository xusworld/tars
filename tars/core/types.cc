#include "tars/core/types.h"

namespace tars {

inline std::string MemcpyKindToString(const MemcpyKind kind) {
  switch (kind) {
    case MemcpyKind::D2D:
      return "Memcpy device2device";
    case MemcpyKind::H2D:
      return "Memcpy host2device";
    case MemcpyKind::D2H:
      return "Memcpy device2host";
    case MemcpyKind::H2H:
      return "Memcpy host2host";
    default:
      return "Memcpy unknown";
  }
}

inline std::string RuntimeTypeToString(const RuntimeType type) {
  switch (type) {
    case RuntimeType::CPU:
      return "Runtime cpu";
    case RuntimeType::CUDA:
      return "Runtime cuda";
    default:
      return "Runtime unknown";
  }
}

}  // namespace tars