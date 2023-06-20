#include <stdint.h>
#include <stdio.h>

#include <fstream>

#include "utils.h"

namespace ace {
namespace converter {

bool OnnxReadProtoFromBinary(const char* filepath,
                             google::protobuf::Message* message) {
  std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    fprintf(stderr, "open failed %s\n", filepath);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  google::protobuf::io::CodedInputStream codedstr(&input);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
  codedstr.SetTotalBytesLimit(INT_MAX);
#else
  codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

  bool success = message->ParseFromCodedStream(&codedstr);

  fs.close();

  return success;
}

bool OnnxWriteProtoFromBinary(const char* filepath,
                              const google::protobuf::Message* message) {
  std::ofstream fs(filepath);
  if (fs.fail()) {
    fprintf(stderr, "open failed %s\n", filepath);
    return false;
  }
  message->SerializeToOstream(&fs);
  fs.close();
  return true;
}

int32_t _limit(int64_t i64) {
  if (i64 > (int64_t)(1 << 30)) {
    return 1 << 30;
  }
  if (i64 < (int64_t)(-(1 << 30))) {
    return (-(1 << 30));
  }
  return i64;
}

}  // namespace converter
}  // namespace ace
