#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(UnaryOnnx);

ace::OpType UnaryOnnx::opType() { return ace::OpType_UnaryOp; }

ace::OpParameter UnaryOnnx::type() { return ace::OpParameter_UnaryOp; }

void UnaryOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  std::unique_ptr<ace::UnaryOpT> unaryOpParam(new ace::UnaryOpT);
  unaryOpParam->T = ace::DataType_DT_FLOAT;

  const auto &originalType = onnxNode->op_type();

#define TO_UNARY_OP(src, dst)   \
  if (originalType == src) {    \
    unaryOpParam->opType = dst; \
  }

  TO_UNARY_OP("Abs", ace::UnaryOpOperation_ABS);
  TO_UNARY_OP("Acos", ace::UnaryOpOperation_ACOS);
  TO_UNARY_OP("Acosh", ace::UnaryOpOperation_ACOSH);
  TO_UNARY_OP("Asinh", ace::UnaryOpOperation_ASINH);
  TO_UNARY_OP("Atan", ace::UnaryOpOperation_ATAN);
  TO_UNARY_OP("Atanh", ace::UnaryOpOperation_ATANH);
  TO_UNARY_OP("Asin", ace::UnaryOpOperation_ASIN);
  TO_UNARY_OP("Ceil", ace::UnaryOpOperation_CEIL);
  TO_UNARY_OP("Cos", ace::UnaryOpOperation_COS);
  TO_UNARY_OP("Cosh", ace::UnaryOpOperation_COSH);
  TO_UNARY_OP("Exp", ace::UnaryOpOperation_EXP);
  TO_UNARY_OP("Erf", ace::UnaryOpOperation_ERF);
  TO_UNARY_OP("Erfc", ace::UnaryOpOperation_ERFC);
  TO_UNARY_OP("Erfinv", ace::UnaryOpOperation_ERFINV);
  TO_UNARY_OP("Expm1", ace::UnaryOpOperation_EXPM1);
  TO_UNARY_OP("Floor", ace::UnaryOpOperation_FLOOR);
  TO_UNARY_OP("HardSwish", ace::UnaryOpOperation_HARDSWISH);
  TO_UNARY_OP("Log", ace::UnaryOpOperation_LOG);
  TO_UNARY_OP("Log1p", ace::UnaryOpOperation_LOG1P);
  TO_UNARY_OP("Gelu", ace::UnaryOpOperation_GELU);
  TO_UNARY_OP("Neg", ace::UnaryOpOperation_NEG);
  TO_UNARY_OP("Sin", ace::UnaryOpOperation_SIN);
  TO_UNARY_OP("Sinh", ace::UnaryOpOperation_SINH);
  TO_UNARY_OP("Sqrt", ace::UnaryOpOperation_SQRT);
  TO_UNARY_OP("Tan", ace::UnaryOpOperation_TAN);
  TO_UNARY_OP("Tanh", ace::UnaryOpOperation_TANH);
  TO_UNARY_OP("Reciprocal", ace::UnaryOpOperation_RECIPROCAL);
  TO_UNARY_OP("Round", ace::UnaryOpOperation_ROUND);
  TO_UNARY_OP("Sign", ace::UnaryOpOperation_SIGN);

  // For specitial error onnx
  TO_UNARY_OP("ATan", ace::UnaryOpOperation_ATAN);
  dstOp->main.value = unaryOpParam.release();
}

REGISTER_CONVERTER(UnaryOnnx, Abs);
REGISTER_CONVERTER(UnaryOnnx, Acos);
REGISTER_CONVERTER(UnaryOnnx, Acosh);
REGISTER_CONVERTER(UnaryOnnx, Asinh);
REGISTER_CONVERTER(UnaryOnnx, Atan);
REGISTER_CONVERTER(UnaryOnnx, Atanh);
REGISTER_CONVERTER(UnaryOnnx, Asin);
REGISTER_CONVERTER(UnaryOnnx, Ceil);
REGISTER_CONVERTER(UnaryOnnx, Cos);
REGISTER_CONVERTER(UnaryOnnx, Cosh);
REGISTER_CONVERTER(UnaryOnnx, Expm1);
REGISTER_CONVERTER(UnaryOnnx, Exp);
REGISTER_CONVERTER(UnaryOnnx, Erf);
REGISTER_CONVERTER(UnaryOnnx, Erfc);
REGISTER_CONVERTER(UnaryOnnx, Erfinv);
REGISTER_CONVERTER(UnaryOnnx, Floor);
REGISTER_CONVERTER(UnaryOnnx, HardSwish);
REGISTER_CONVERTER(UnaryOnnx, Log);
REGISTER_CONVERTER(UnaryOnnx, Log1p);
REGISTER_CONVERTER(UnaryOnnx, Gelu);
REGISTER_CONVERTER(UnaryOnnx, Neg);
REGISTER_CONVERTER(UnaryOnnx, Sin);
REGISTER_CONVERTER(UnaryOnnx, Tan);
REGISTER_CONVERTER(UnaryOnnx, Tanh);
REGISTER_CONVERTER(UnaryOnnx, Reciprocal);
REGISTER_CONVERTER(UnaryOnnx, Round);
REGISTER_CONVERTER(UnaryOnnx, Sign);
REGISTER_CONVERTER(UnaryOnnx, Sinh);
REGISTER_CONVERTER(UnaryOnnx, Sqrt);

// For specitial error onnx
REGISTER_CONVERTER(UnaryOnnx, ATan);

}  // namespace converter
}  // namespace ace