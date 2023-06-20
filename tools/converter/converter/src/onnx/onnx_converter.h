#pragma once

#include <flatbuffers/idl.h>
#include <flatbuffers/minireflect.h>
#include <flatbuffers/util.h>

#include <iostream>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/proto/onnx.pb.h"

namespace ace {
namespace converter {

int Onnx2AceNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<ace::NetT> &netT);

}
}  // namespace ace