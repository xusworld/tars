#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ace {
namespace converter {

enum class FrontendFramework {
  ONNX = 0,
  TENSORFLOW,
  PyTorch,
  CAFFE,
};

class OptimizeConf final {
 public:
  OptimizeConf() = default;

  int optimize_level;

  bool is_half_mode;
  int weightQuantBits = 0;  // If weightQuantBits > 0, it means the bit
  bool weightQuantAsymmetric = false;
  // The path of the model compression file that stores the int8 calibration
  // table or sparse parameters.
  std::string compressionParamsFile = "";
  int optimizePrefer = 0;
};

class BuildOption final {
  FrontendFramework frontend_framework;
  std::string source_model_path;
  std::string saved_model_path;
  std::string input_data_layout;

  std::vector<std::string> input_names;
  std::vector<std::vector<int>> input_shapes;
  std::vector<std::string> output_names;
  std::vector<std::vector<int>> output_shapes;

 private:
  std::string ace_ver_;
  std::unordered_map<std::string, std::vector<int>> name2shape_;
  OptimizeConf optimize_conf_;
};

}  // namespace converter
}  // namespace ace
