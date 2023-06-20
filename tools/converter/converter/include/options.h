#pragma once

#include <string>

namespace common {

typedef struct Options {
  bool doCompress;
} Options;

Options DefaultOptions();

Options BuildOptions(const std::string& compressionFile);

}  // namespace common
