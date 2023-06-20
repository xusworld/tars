#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

// ======================= SplitToSequence =========================
DECLARE_OP_CONVERTER(SplitToSequenceOnnx);

ace::OpType SplitToSequenceOnnx::opType() {
  return ace::OpType_TensorArraySplit;
}
ace::OpParameter SplitToSequenceOnnx::type() {
  return ace::OpParameter_TensorArray;
}

void SplitToSequenceOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                              OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();

    if (attributeName == "axis") {
      tensorArray->axis = attributeProto.i();
    }
    if (attributeName == "keepdims") {
      tensorArray->keepdims = attributeProto.i();
    }
  }
  dstOp->main.value = tensorArray;

  // split(optional) -> split(required), because ace Size Computer need explicit
  // split index
  if (dstOp->inputIndexes.size() == 1) {
    dstOp->inputIndexes.push_back(
        scope->buildIntConstOp({1}, dstOp->name + "/split_default"));
  }
  auto tensorArrayIdx =
      scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
  int valueIdx = dstOp->inputIndexes[0];
  int splitIdx = dstOp->inputIndexes[1];
  dstOp->inputIndexes.resize(4);
  // handle, value, lengths, flow_in
  dstOp->inputIndexes[0] = tensorArrayIdx.first;
  dstOp->inputIndexes[1] = valueIdx;
  dstOp->inputIndexes[2] = splitIdx;
  dstOp->inputIndexes[3] = tensorArrayIdx.second;
}

REGISTER_CONVERTER(SplitToSequenceOnnx, SplitToSequence);

// ======================= SequenceAt =========================
DECLARE_OP_CONVERTER(SequenceAtOnnx);
ace::OpType SequenceAtOnnx::opType() { return ace::OpType_TensorArrayRead; }
ace::OpParameter SequenceAtOnnx::type() { return ace::OpParameter_TensorArray; }
void SequenceAtOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;
  // handle, index, flow_in and handle == flow_in
  dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(SequenceAtOnnx, SequenceAt);

// ======================= SequenceLength =========================
DECLARE_OP_CONVERTER(SequenceLengthOnnx);
ace::OpType SequenceLengthOnnx::opType() { return ace::OpType_TensorArraySize; }
ace::OpParameter SequenceLengthOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void SequenceLengthOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                             OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;
  // handle, flow_in and handle == flow_in
  dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(SequenceLengthOnnx, SequenceLength);

// ======================= SequenceInsert =========================
DECLARE_OP_CONVERTER(SequenceInsertOnnx);
ace::OpType SequenceInsertOnnx::opType() {
  return ace::OpType_TensorArrayInsert;
}
ace::OpParameter SequenceInsertOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void SequenceInsertOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                             OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;

  auto& indexs = dstOp->inputIndexes;
  // position(optional) -> position(required), because ace SizeComputer need
  // explcit position index
  if (indexs.size() == 2) {
    std::unique_ptr<ace::OpT> op(new ace::OpT);
    auto name = dstOp->name + "/seq_length";
    op->name = name;
    op->type = ace::OpType_TensorArraySize;
    op->inputIndexes.push_back(op->inputIndexes[0]);
    op->outputIndexes.push_back(scope->declareTensor(name));
    indexs.push_back(op->outputIndexes[0]);
    scope->oplists().emplace_back(std::move(op));
  }
  // handle, tensor, position => handle, position, tensor, flow_in, for reusing
  // inference code of OpType_TensorArrayWrite
  std::swap(indexs[1], indexs[2]);
  indexs.push_back(indexs[0]);
}
REGISTER_CONVERTER(SequenceInsertOnnx, SequenceInsert);

// ======================= SequenceErase =========================
DECLARE_OP_CONVERTER(SequenceEraseOnnx);
ace::OpType SequenceEraseOnnx::opType() { return ace::OpType_TensorArrayErase; }
ace::OpParameter SequenceEraseOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void SequenceEraseOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                            OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;

  auto& indexs = dstOp->inputIndexes;
  // position(optional) -> position(required), because ace SizeComputer need
  // explcit position index
  if (indexs.size() == 1) {
    // onnx default erase last element of seq, so index = seq.size() - 1
    std::unique_ptr<ace::OpT> opSize(new ace::OpT), opSub(new ace::OpT);
    auto nameSize = dstOp->name + "/seq_length",
         nameSub = dstOp->name + "/seq_index";
    auto idxSize = scope->declareTensor(nameSize),
         idxSub = scope->declareTensor(nameSub);
    opSize->name = nameSize;
    opSize->type = ace::OpType_TensorArraySize;
    opSize->inputIndexes.assign(2, indexs[0]);  // handle, flow_in
    opSize->outputIndexes.push_back(idxSize);
    opSub->name = nameSub;
    opSub->type = ace::OpType_BinaryOp;
    opSub->main.type = ace::OpParameter_BinaryOp;
    auto paramSub = new ace::BinaryOpT;
    paramSub->opType = ace::BinaryOpOperation_SUB;
    opSub->main.value = paramSub;
    opSub->inputIndexes.assign(
        {idxSize, scope->buildIntConstOp({1}, dstOp->name + "/const")});
    opSub->outputIndexes.push_back(idxSub);
    scope->oplists().emplace_back(std::move(opSize));
    scope->oplists().emplace_back(std::move(opSub));
    indexs.push_back(idxSub);
  }
  indexs.push_back(indexs[0]);
}
REGISTER_CONVERTER(SequenceEraseOnnx, SequenceErase);

// ======================= ConcatFromSequence =========================
DECLARE_OP_CONVERTER(ConcatFromSequenceOnnx);
ace::OpType ConcatFromSequenceOnnx::opType() {
  return ace::OpType_TensorArrayConcat;
}
ace::OpParameter ConcatFromSequenceOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void ConcatFromSequenceOnnx::run(ace::OpT* dstOp,
                                 const onnx::NodeProto* onnxNode,
                                 OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();

    if (attributeName == "axis") {
      tensorArray->axis = attributeProto.i();
    }
    if (attributeName == "new_axis") {
      tensorArray->new_axis = attributeProto.i();
    }
  }
  dstOp->main.value = tensorArray;
  // handle, flow_in and handle == flow_in
  dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(ConcatFromSequenceOnnx, ConcatFromSequence);

// ======================= SequenceConstruct =========================
DECLARE_OP_CONVERTER(SequenceConstructOnnx);
ace::OpType SequenceConstructOnnx::opType() {
  return ace::OpType_TensorArrayWrite;
}
ace::OpParameter SequenceConstructOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void SequenceConstructOnnx::run(ace::OpT* dstOp,
                                const onnx::NodeProto* onnxNode,
                                OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;

  auto tensorArrayIdx =
      scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
  int inputNum = dstOp->inputIndexes.size();
  for (int i = 0; i < inputNum; ++i) {
    auto op = (i == inputNum - 1 ? dstOp : new ace::OpT);
    int insert_idx = scope->buildIntConstOp(
        {i}, dstOp->name + "/tmp_index_" + std::to_string(i));
    int value_idx = dstOp->inputIndexes[i];
    op->inputIndexes.assign(
        {tensorArrayIdx.first, insert_idx, value_idx, tensorArrayIdx.second});
    if (i < inputNum - 1) {
      auto name = dstOp->name + "/tmp_seq_" + std::to_string(i);
      op->name = name;
      op->type = ace::OpType_TensorArrayWrite;
      op->main.type = ace::OpParameter_TensorArray;
      auto tensorArray = new ace::TensorArrayT;
      tensorArray->T = ace::DataType_DT_FLOAT;
      op->main.value = tensorArray;
      int output_idx = scope->declareTensor(name);
      op->outputIndexes.assign({output_idx});
      scope->oplists().emplace_back(op);
      tensorArrayIdx.first = tensorArrayIdx.second = output_idx;
    }
  }
}
REGISTER_CONVERTER(SequenceConstructOnnx, SequenceConstruct);

// ======================= SequenceEmpty =========================
DECLARE_OP_CONVERTER(SequenceEmptyOnnx);
ace::OpType SequenceEmptyOnnx::opType() { return ace::OpType_TensorArray; }
ace::OpParameter SequenceEmptyOnnx::type() {
  return ace::OpParameter_TensorArray;
}
void SequenceEmptyOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                            OnnxScope* scope) {
  auto tensorArray = new ace::TensorArrayT;
  tensorArray->T = ace::DataType_DT_FLOAT;
  dstOp->main.value = tensorArray;

  auto tensorArrayIdx =
      scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
  scope->oplists()[scope->oplists().size() - 2]->main.AsBlob()->int32s[0] =
      0;  // change init_size 1 -> 0

  dstOp->inputIndexes.resize(2);
  dstOp->inputIndexes[0] = tensorArrayIdx.first;
  dstOp->inputIndexes[1] = tensorArrayIdx.second;
}
REGISTER_CONVERTER(SequenceEmptyOnnx, SequenceEmpty);

// ======================= ReverseSequence =========================
DECLARE_OP_CONVERTER(ReverseSequenceOnnx);
ace::OpType ReverseSequenceOnnx::opType() {
  return ace::OpType_ReverseSequence;
}
ace::OpParameter ReverseSequenceOnnx::type() {
  return ace::OpParameter_ReverseSequenceParam;
}
void ReverseSequenceOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                              OnnxScope* scope) {
  int batchDim = 1, seqDim = 0;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "batch_axis") {
      batchDim = attributeProto.i();
    }
    if (attributeName == "time_axis") {
      seqDim = attributeProto.i();
    }
  }
  auto param = new ace::ReverseSequenceParamT;
  param->batchDim = batchDim;
  param->seqDim = seqDim;
  dstOp->main.value = param;
}
REGISTER_CONVERTER(ReverseSequenceOnnx, ReverseSequence);

}  // namespace converter
}  // namespace ace