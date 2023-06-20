#include "default_onnx_op_converter.h"
#include "include/op_count.h"
#include "onnx_op_converter_register.h"

namespace ace {
namespace converter {

OnnxOpConverterSuit::OnnxOpConverterSuit() {}

OnnxOpConverterSuit::~OnnxOpConverterSuit() {
  for (auto& iter : mConverterContainer) {
    delete iter.second;
  }
  mConverterContainer.clear();
}

OnnxOpConverterSuit* OnnxOpConverterSuit::global = nullptr;

OnnxOpConverterSuit* OnnxOpConverterSuit::get() {
  if (global == nullptr) {
    global = new OnnxOpConverterSuit;
  }
  return global;
}

void OnnxOpConverterSuit::insert(OnnxOpConverter* t, const char* name) {
  ace::OpCount::get()->insertOp("ONNX", std::string(name));
  mConverterContainer.insert(std::make_pair(name, t));
}

OnnxOpConverter* OnnxOpConverterSuit::search(const std::string& name) {
  auto iter = mConverterContainer.find(name);
  if (iter == mConverterContainer.end()) {
    static DefaultOnnxOpConverter defaultConverter;
    return &defaultConverter;
  }
  return iter->second;
}
}  // namespace converter
}  // namespace ace