#pragma once

#include <string>

namespace ace {

enum class MemcpyKind {
  H2D,  // host to device
  D2H,  // device to host
  D2D,  // device to device
  H2H,  // host to host
};

enum class RuntimeType { CPU = 0, CUDA = 1 };

// returns the memcpy kind string
std::string MemcpyKindToString(const MemcpyKind kind);

// returns the runtime type string
std::string RuntimeTypeToString(const RuntimeType type);
}  // namespace ace