#include <numeric>
#include <sstream>

#include "tars/core/utils.h"

namespace tars {

// inline int DataType2Bytes(const DataType dtype) {
//   switch (dtype) {
//     case DataType_NONE:
//       return 0;
//     case DataType_INT_8:
//       return 1;
//     case DataType_INT_16:
//       return 2;
//     case DataType_INT_32:
//       return 4;
//     case DataType_INT_64:
//       return 8;
//     case DataType_UINT_8:
//       return 1;
//     case DataType_UINT_16:
//       return 2;
//     case DataType_UINT_32:
//       return 4;
//     case DataType_UINT_64:
//       return 8;
//     case DataType_FLOAT_16:
//       return 2;
//     case DataType_FLOAT_32:
//       return 4;
//     case DataType_FLOAT_64:
//       return 8;
//       break;
//   }

//   return 0;
// }

int DataType2Bytes(const DataType dtype) {
  switch (dtype) {
    // case none:
    //   return 0;
    // case int8:
    //   return 1;
    // case int16:
    //   return 2;
    // case int32:
    //   return 4;
    // case int64:
    //   return 8;
    // case uint8:
    //   return 1;
    // case uint16:
    //   return 2;
    // case uint32:
    //   return 4;
    // case uint64:
    //   return 8;
    // case DataType_DT_FLOAT:
    //   return 2;
    case DataType_DT_FLOAT:
      return 4;
      // case DataType_FLOAT_64:
      //   return 8;
      break;
  }

  return 0;
}

inline std::string DataType2String(const DataType dtype) {
  // switch (dtype) {
  //   case DataType_NONE:
  //     return "none";
  //   case DataType_INT_8:
  //     return "int8";
  //   case DataType_INT_16:
  //     return "int16";
  //   case DataType_INT_32:
  //     return "int32";
  //   case DataType_INT_64:
  //     return "int64";
  //   case DataType_UINT_8:
  //     return "uint8";
  //   case DataType_UINT_16:
  //     return "uint16";
  //   case DataType_UINT_32:
  //     return "uint32";
  //   case DataType_UINT_64:
  //     return "uint64";
  //   case DataType_FLOAT_16:
  //     return "half";
  //   case DataType_FLOAT_32:
  //     return "float";
  //   case DataType_FLOAT_64:
  //     return "double";
  //     break;
  // }

  return "";
}

int32_t GetBufferBytes(const DataType dtype, const int size) {
  return size * DataType2Bytes(dtype);
}

int32_t GetBufferBytes(const DataType dtype, const std::vector<int32_t>& dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return size * DataType2Bytes(dtype);
}

}  // namespace tars