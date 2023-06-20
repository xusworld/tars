#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(BinaryOpOnnx);

ace::OpType BinaryOpOnnx::opType() { return ace::OpType_BinaryOp; }

ace::OpParameter BinaryOpOnnx::type() { return ace::OpParameter_BinaryOp; }

void BinaryOpOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                       OnnxScope* scope) {
  const auto& originalType = onnxNode->op_type();
  int inputSize = onnxNode->input_size();
  if (inputSize == 1) {
    DLOG(FATAL) << "Not support 1 input for " << originalType << " op";
    return;
  }
  std::vector<std::string> moreOps = {"Max", "Min", "Sum", "Mean"};
  if (inputSize > 2 && std::find(moreOps.begin(), moreOps.end(),
                                 originalType) == moreOps.end()) {
    DLOG(FATAL) << "Not support more than 2 input for " << originalType
                << " op";
    return;
  }

  void* param;
  if (inputSize == 2) {
    param = new ace::BinaryOpT;
  } else {
    param = new ace::ReductionParamT;
  }
#define TO_BINARY_OP(src, dst)              \
  if (originalType == src) {                \
    ((ace::BinaryOpT*)param)->opType = dst; \
  }
#define TO_REDUCE_OP(src, dst)                       \
  if (originalType == src) {                         \
    ((ace::ReductionParamT*)param)->operation = dst; \
  }
#define TO_BINARY_OR_REDUCE_OP(src, dst0, dst1)         \
  if (originalType == src) {                            \
    if (inputSize == 2) {                               \
      ((ace::BinaryOpT*)param)->opType = dst0;          \
    } else {                                            \
      ((ace::ReductionParamT*)param)->operation = dst1; \
    }                                                   \
  }

  TO_BINARY_OP("Add", ace::BinaryOpOperation_ADD);
  TO_BINARY_OP("And", ace::BinaryOpOperation_MUL);
  TO_BINARY_OP("Div", ace::BinaryOpOperation_REALDIV);
  TO_BINARY_OP("Mul", ace::BinaryOpOperation_MUL);
  TO_BINARY_OP("Equal", ace::BinaryOpOperation_EQUAL);
  TO_BINARY_OP("Less", ace::BinaryOpOperation_LESS);
  TO_BINARY_OP("LessOrEqual", ace::BinaryOpOperation_LESS_EQUAL);
  TO_BINARY_OP("Greater", ace::BinaryOpOperation_GREATER);
  TO_BINARY_OP("GreaterOrEqual", ace::BinaryOpOperation_GREATER_EQUAL);
  TO_BINARY_OR_REDUCE_OP("Max", ace::BinaryOpOperation_MAXIMUM,
                         ace::ReductionType_MAXIMUM);
  TO_BINARY_OR_REDUCE_OP("Min", ace::BinaryOpOperation_MINIMUM,
                         ace::ReductionType_MINIMUM);
  if (originalType == "Mod") {
    int fmod = 0;
    for (const auto& attrProto : onnxNode->attribute()) {
      if (attrProto.name() == "fmod") {
        fmod = attrProto.i();
      }
    }
    ((ace::BinaryOpT*)param)->opType =
        (fmod == 0 ? ace::BinaryOpOperation_MOD
                   : ace::BinaryOpOperation_FLOORMOD);
  }
  TO_BINARY_OP("Pow", ace::BinaryOpOperation_POW);
  TO_BINARY_OP("Sub", ace::BinaryOpOperation_SUB);
  TO_BINARY_OR_REDUCE_OP("Sum", ace::BinaryOpOperation_ADD,
                         ace::ReductionType_SUM);
  TO_REDUCE_OP("Mean", ace::ReductionType_MEAN);
  TO_BINARY_OP("Or", ace::BinaryOpOperation_LOGICALOR);
  TO_BINARY_OP("Xor", ace::BinaryOpOperation_LOGICALXOR);

  if (originalType == "BitShift") {
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
      const auto& attributeProto = onnxNode->attribute(i);
      const auto& attributeName = attributeProto.name();
      if (attributeName == "direction") {
        if (attributeProto.s() == "LEFT") {
          ((ace::BinaryOpT*)param)->opType = ace::BinaryOpOperation_LEFTSHIFT;
        } else {
          ((ace::BinaryOpT*)param)->opType = ace::BinaryOpOperation_RIGHTSHIFT;
        }
      }
    }
  }

  if (inputSize == 2) {
    dstOp->main.value = param;
    return;
  }

  // N input (i0, i1, ...) => 1 input (N, i0, i1, ...)
  std::unique_ptr<ace::OpT> pack(new ace::OpT);
  auto packName = dstOp->name + "/packed_input";
  pack->name = packName;
  pack->type = ace::OpType_Pack;
  pack->main.type = ace::OpParameter_PackParam;
  pack->main.value = new ace::PackParamT;
  pack->main.AsPackParam()->axis = 0;
  pack->inputIndexes = dstOp->inputIndexes;
  int packedInput = scope->declareTensor(packName);
  pack->outputIndexes.assign({packedInput});
  scope->oplists().emplace_back(std::move(pack));

  // Reduce(Max/Min/Sum/Mean) along axis 0
  dstOp->type = ace::OpType_Reduction;
  dstOp->main.type = ace::OpParameter_ReductionParam;
  ((ace::ReductionParamT*)param)->dim.assign({0});
  dstOp->main.value = param;
  dstOp->main.AsReductionParam()->keepDims = false;
  dstOp->main.AsReductionParam()->dim.assign({0});
  dstOp->inputIndexes.assign({packedInput});
}

REGISTER_CONVERTER(BinaryOpOnnx, Add);
REGISTER_CONVERTER(BinaryOpOnnx, And);
REGISTER_CONVERTER(BinaryOpOnnx, Sum);
REGISTER_CONVERTER(BinaryOpOnnx, Sub);
REGISTER_CONVERTER(BinaryOpOnnx, Div);
REGISTER_CONVERTER(BinaryOpOnnx, Mul);
REGISTER_CONVERTER(BinaryOpOnnx, Pow);
REGISTER_CONVERTER(BinaryOpOnnx, Equal);
REGISTER_CONVERTER(BinaryOpOnnx, Less);
REGISTER_CONVERTER(BinaryOpOnnx, LessOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Greater);
REGISTER_CONVERTER(BinaryOpOnnx, GreaterOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Max);
REGISTER_CONVERTER(BinaryOpOnnx, Min);
REGISTER_CONVERTER(BinaryOpOnnx, Mod);
REGISTER_CONVERTER(BinaryOpOnnx, Or);
REGISTER_CONVERTER(BinaryOpOnnx, Xor);
REGISTER_CONVERTER(BinaryOpOnnx, BitShift);
REGISTER_CONVERTER(BinaryOpOnnx, Mean);

}  // namespace converter
}  // namespace ace