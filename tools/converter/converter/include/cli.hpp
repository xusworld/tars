#pragma once

#include <iostream>

#include "config.h"

namespace cae {

class Cli {
 public:
  static bool initializeMNNConvertArgs(AceModelConfig& modelPath, int argc,
                                       char** argv);
  static bool convertModel(AceModelConfig& modelPath);
  static int testconvert(const std::string& defaultCacheFile,
                         const std::string& directName, float maxErrorRate);
  static bool mnn2json(const char* modelFile, const char* jsonFile,
                       int flag = 3);
  static bool json2mnn(const char* jsonFile, const char* modelFile);
};

};  // namespace cae

class CommonKit {
 public:
  static bool FileIsExist(std::string path);
};
