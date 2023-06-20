#pragma once

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace ace {
namespace converter {

bool OnnxReadProtoFromBinary(const char* filepath,
                             google::protobuf::Message* message);

bool OnnxWriteProtoFromBinary(const char* filepath,
                              const google::protobuf::Message* message);

int32_t _limit(int64_t i64);

}  // namespace converter
}  // namespace ace
