#include <glog/logging.h>

#include "ace/schema/type_generated.h"
#include "include/op_count.h"
#include "onnx_op_converter.h"
#include "src/onnx/proto/onnx.pb.h"
#include "utils.h"

namespace ace {
namespace converter {

ace::DataType OnnxOpConverter::convertDataType(int32_t dtype) {
  switch (dtype) {
    case onnx::TensorProto_DataType_INT8:
      return ace::DataType_DT_INT8;
    case onnx::TensorProto_DataType_UINT8:
      return ace::DataType_DT_UINT8;
    case onnx::TensorProto_DataType_INT32:
      return ace::DataType_DT_INT32;
    case onnx::TensorProto_DataType_INT64:
      return ace::DataType_DT_INT64;
    case onnx::TensorProto_DataType_FLOAT:
      return ace::DataType_DT_FLOAT;
    case onnx::TensorProto_DataType_DOUBLE:
      return ace::DataType_DT_DOUBLE;
  }
  return ace::DataType_DT_INVALID;
}

ace::BlobT* OnnxOpConverter::convertTensorToBlob(
    const onnx::TensorProto* constantTp, const std::string& modelDir) {
  auto constantParam = new ace::BlobT;
  auto dataType = convertDataType(constantTp->data_type());
  // printf("origindataType = %d, dataType = %s\n", constantTp->data_type(),
  // ace::EnumNameDataType(dataType));

  constantParam->dataType = dataType;
  constantParam->dataFormat = ace::DataFormat_NCHW;

  size_t dimSize = constantTp->dims().size();
  constantParam->dims.resize(dimSize);
  size_t dataSize = 1;
  for (int i = 0; i < dimSize; ++i) {
    constantParam->dims[i] = constantTp->dims(i);
    dataSize = dataSize * constantTp->dims(i);
  }
  std::vector<int64_t> alignContent;
  if (constantTp->data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    std::string location;
    int64_t offset = 0;
    int64_t length = -1;
    for (const auto& k : constantTp->external_data()) {
      if (k.key() == "location") {
        location = k.value();
      } else if (k.key() == "offset") {
        offset = std::atoll(k.value().c_str());
      } else if (k.key() == "length") {
        length = std::atoll(k.value().c_str());
      }
    }
    if (!modelDir.empty()) {
      location = modelDir + location;
    }

    auto fp = fopen(location.c_str(), "rb");
    if (fp == nullptr) {
      LOG(FATAL) << "Fail to open external data: " << location;
      return nullptr;
    }
    if (length < 0) {
      fseek(fp, 0, SEEK_END);
      length = ftell(fp) - offset;
    }
    fseek(fp, offset, SEEK_SET);
    alignContent.resize((length + sizeof(int64_t) - 1) / sizeof(int64_t));
    fread(alignContent.data(), 1, length, fp);
    fclose(fp);
  } else {
    alignContent.resize((constantTp->raw_data().size() + sizeof(int64_t) - 1) /
                        sizeof(int64_t));
    ::memcpy(alignContent.data(), constantTp->raw_data().data(),
             constantTp->raw_data().size());
  }

  const void* tensor_content = (const void*)alignContent.data();

  switch (constantTp->data_type()) {
#define CASE_DATA_TYPE(src, dst)                        \
  case src:                                             \
    if (constantTp->dst##_data_size() != 0) {           \
      tensor_content = constantTp->dst##_data().data(); \
    }                                                   \
    break;
    CASE_DATA_TYPE(onnx::TensorProto_DataType_DOUBLE, double);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_INT64, int64);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_INT32, int32);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_UINT64, uint64);
    CASE_DATA_TYPE(onnx::TensorProto_DataType_BOOL, int32);
    default:
      break;
  }
  if (0 == dataSize) {
    // Empty blob
    return constantParam;
  }

  if (!tensor_content) {
    DLOG(FATAL) << "Convert no data, "
                   "Please make sure ";
    return nullptr;
  }

  switch (constantTp->data_type()) {
    case onnx::TensorProto_DataType_DOUBLE: {
      constantParam->float32s.resize(dataSize);
      auto source = (double*)tensor_content;

      for (int i = 0; i < dataSize; ++i) {
        constantParam->float32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      constantParam->int32s.resize(dataSize);
      auto source = (int64_t*)tensor_content;

      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = _limit(source[i]);
      }
      break;
    }
    case onnx::TensorProto_DataType_INT32: {
      auto source = (int32_t*)tensor_content;
      constantParam->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT16: {
      auto source = (uint16_t*)tensor_content;
      constantParam->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT16: {
      auto source = (int16_t*)tensor_content;
      constantParam->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {
      auto source = (bool*)tensor_content;
      constantParam->int32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_INT8: {
      auto source = (int8_t*)tensor_content;
      constantParam->int8s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int8s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT8: {
      auto source = (uint8_t*)tensor_content;
      constantParam->uint8s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->uint8s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      float* tempFloatData = (float*)tensor_content;
      constantParam->float32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->float32s[i] = tempFloatData[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT32: {
      auto source = (uint32_t*)tensor_content;
      constantParam->float32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {
      auto source = (uint64_t*)tensor_content;
      constantParam->float32s.resize(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        constantParam->int32s[i] = source[i];
      }
      break;
    }
    default: {
      DLOG(FATAL) << "Don't support " << constantTp->data_type();
      break;
    }
  }
  return constantParam;
}

}  // namespace converter
}  // namespace ace