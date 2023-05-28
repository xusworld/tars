#pragma once

#include "ir/current/Type_generated.h"
#include "tars/core/macro.h"

namespace tars {

// DataType to bytes.
int DataType2Bytes(const DataType dtype);

// DataType to string.
std::string DataType2String(const DataType dtype);

int32_t GetBufferBytes(const DataType dtype, const int size);
int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims);

template <typename T>
inline T DivDown(T n, T div) {
  return n / div * div;
}

template <typename T>
inline T DivUp(T n, T div) {
  return (n + div - 1) / div * div;
}

// Math
#ifndef UP_DIV
#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif
#ifndef ALIGN_UP4
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#endif
#ifndef ALIGN_UP8
#define ALIGN_UP8(x) ROUND_UP((x), 8)
#endif
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

}  // namespace tars