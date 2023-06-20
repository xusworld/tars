#include "default_onnx_op_converter.h"

namespace ace {
namespace converter {

void DefaultOnnxOpConverter::run(ace::OpT* dstOp,
                                 const onnx::NodeProto* onnxNode,
                                 OnnxScope* scope) {
  auto extra = new ExtraT;
  dstOp->main.type = OpParameter_Extra;
  dstOp->main.value = extra;
  extra->engine = "ONNX";
  extra->type = onnxNode->op_type();
  for (auto srcAttr : onnxNode->attribute()) {
    std::unique_ptr<AttributeT> attr(new AttributeT);
    attr->key = srcAttr.name();
    switch (srcAttr.type()) {
      case onnx::AttributeProto_AttributeType_INTS:
        attr->list.reset(new ListValueT);
        attr->list->i.resize(srcAttr.ints_size());
        for (int i = 0; i < srcAttr.ints_size(); ++i) {
          attr->list->i[i] = _limit(srcAttr.ints(i));
        }
        break;
      case onnx::AttributeProto_AttributeType_FLOATS:
        attr->list.reset(new ListValueT);
        attr->list->f.resize(srcAttr.floats_size());
        for (int i = 0; i < srcAttr.floats_size(); ++i) {
          attr->list->f[i] = srcAttr.floats(i);
        }
        break;
      case onnx::AttributeProto_AttributeType_TENSOR:
        attr->tensor.reset(convertTensorToBlob(&srcAttr.t()));
        break;
      default:
        break;
    }
    attr->i = _limit(srcAttr.i());
    attr->s = srcAttr.s();
    attr->f = srcAttr.f();
    extra->attr.emplace_back(std::move(attr));
  }
}

}  // namespace converter
}  // namespace ace