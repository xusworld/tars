#pragma once

#include "src/onnx/onnx_op_converter.h"

namespace ace {
namespace converter {

class OnnxOpConverterSuit {
 public:
  OnnxOpConverterSuit();
  ~OnnxOpConverterSuit();

 public:
  static OnnxOpConverterSuit* get();
  void insert(OnnxOpConverter* t, const char* name);
  OnnxOpConverter* search(const std::string& name);

 private:
  static OnnxOpConverterSuit* global;
  std::map<std::string, OnnxOpConverter*> mConverterContainer;
};

template <typename T>
class OnnxOpConverterRegister {
 public:
  OnnxOpConverterRegister(const char* name) {
    T* opConverter = new T;
    OnnxOpConverterSuit* container = OnnxOpConverterSuit::get();
    container->insert(opConverter, name);
  }
  ~OnnxOpConverterRegister() {}

 private:
  OnnxOpConverterRegister();
};

#define DECLARE_OP_CONVERTER(name)                                     \
  class name : public OnnxOpConverter {                                \
   public:                                                             \
    name() {}                                                          \
    virtual ~name() {}                                                 \
    virtual void run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode, \
                     OnnxScope* scope);                                \
    virtual ace::OpType opType();                                      \
    virtual ace::OpParameter type();                                   \
  }
#define REGISTER_CONVERTER(name, opType) \
  static OnnxOpConverterRegister<name> _Convert_##opType(#opType)

}  // namespace converter
}  // namespace ace