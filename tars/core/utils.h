#pragma once

#include "tars/core/macro.h"

namespace ace {

// DataType to bytes.
int DataType2Bytes(const DataType dtype);

// DataType to string.
std::string DataType2String(const DataType dtype);

int32_t GetBufferBytes(const DataType dtype, const int size);
int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims);

}  // namespace ace