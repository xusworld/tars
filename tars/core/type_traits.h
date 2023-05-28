#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include "ir/current/Type_generated.h"
#include "tars/core/macro.h"
#include "tars/core/status.h"

namespace tars {

// template <size_t index, typename Arg, typename... Args>
// struct ParamPackType;

// template <size_t index, typename Arg, typename... Args>
// struct ParamPackType : ParamPackType<index - 1, Args...> {};

// template <typename Arg, typename... Args>
// struct ParamPackType<0, Arg, Args...> {
//   typedef Arg type;
// };

// template <typename T>
// struct function_traits;

// template <typename RetType, typename... Args>
// struct function_traits<RetType(Args...)> {
//   typedef RetType return_type;
//   enum { size = sizeof...(Args) };

//   template <size_t index>
//   struct Param {
//     typedef typename ParamPackType<index, Args...>::type type;
//   };
// };

// template <typename ClassType, typename RetType, typename... Args>
// struct function_traits<RetType (ClassType::*)(Args...) const> {
//   typedef RetType return_type;
//   enum { size = sizeof...(Args) };

//   template <size_t index>
//   struct Param {
//     typedef typename ParamPackType<index, Args...>::type type;
//   };
// };

// template <typename LambdaFunc>
// struct function_traits : function_traits<decltype(&LambdaFunc::operator())>
// {};

// template <typename RetType, typename... Args>
// struct function_traits<RetType(Args...) const>
//     : function_traits<RetType(Args...)> {};

/// Judge if the function return type is void.
// template <typename>
// struct is_void_function;

// template <typename functor>
// struct is_void_function
//     : std::is_void<typename function_traits<functor>::return_type> {};

// /// Judge if the function return type is Status.
// template <typename>
// struct is_status_function;

// template <typename functor>
// struct is_status_function
//     : std::is_same<typename function_traits<functor>::return_type, Status>
//     {};

/// Type changing for  std::vector<bool> which considered a mistake in STL.
template <typename T>
struct std_vector_type_warpper {
  typedef T type;
  typedef T ret_type;
};

template <>
struct std_vector_type_warpper<bool> {
  typedef std::string type;
  typedef bool ret_type;
};

template <>
struct std_vector_type_warpper<const bool> {
  typedef std::string type;
  typedef const bool ret_type;
};

template <typename T>
struct is_bool_type : std::is_same<T, bool> {};

template <typename T>
struct is_bool_type<const T> : std::is_same<T, bool> {};

template <bool Boolean>
struct Bool2Type {};

template <size_t index, typename Arg, typename... Args>
struct ParamPackType;

template <size_t index, typename Arg, typename... Args>
struct ParamPackType : ParamPackType<index - 1, Args...> {};

template <typename Arg, typename... Args>
struct ParamPackType<0, Arg, Args...> {
  typedef Arg type;
};

template <typename T>
struct function_traits;

template <typename RetType, typename... Args>
struct function_traits<RetType(Args...)> {
  typedef RetType return_type;
  enum { size = sizeof...(Args) };

  template <size_t index>
  struct Param {
    typedef typename ParamPackType<index, Args...>::type type;
  };
};

template <typename ClassType, typename RetType, typename... Args>
struct function_traits<RetType (ClassType::*)(Args...) const> {
  typedef RetType return_type;
  enum { size = sizeof...(Args) };

  template <size_t index>
  struct Param {
    typedef typename ParamPackType<index, Args...>::type type;
  };
};

template <typename LambdaFunc>
struct function_traits : function_traits<decltype(&LambdaFunc::operator())> {};

template <typename RetType, typename... Args>
struct function_traits<RetType(Args...) const>
    : function_traits<RetType(Args...)> {};

/// Judge if the function return type is void.
template <typename>
struct is_void_function;

template <typename functor>
struct is_void_function
    : std::is_void<typename function_traits<functor>::return_type> {};

/// Judge if the function return type is Status.
template <typename>
struct is_status_function;

template <typename functor>
struct is_status_function
    : std::is_same<typename function_traits<functor>::return_type, Status> {};

struct __invalid_type {};

template <DataType dtype>
struct DataTypeTraits {
  typedef __invalid_type value_type;
  typedef __invalid_type reference;
  typedef __invalid_type const_reference;
  typedef __invalid_type pointer;
  typedef __invalid_type const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_INT8> {
  typedef int8_t value_type;
  typedef int8_t& reference;
  typedef const int8_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_INT16> {
  typedef int16_t value_type;
  typedef int16_t& reference;
  typedef const int16_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_INT32> {
  typedef int32_t value_type;
  typedef int32_t& reference;
  typedef const int32_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_INT64> {
  typedef int64_t value_type;
  typedef int64_t& reference;
  typedef const int64_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_UINT8> {
  typedef uint8_t value_type;
  typedef uint8_t& reference;
  typedef const uint8_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

template <>
struct DataTypeTraits<DataType_DT_UINT16> {
  typedef uint16_t value_type;
  typedef uint16_t& reference;
  typedef const uint16_t& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

// template <>
// struct DataTypeTraits<DataType_DT_UINT32> {
//   typedef uint32_t value_type;
//   typedef uint32_t& reference;
//   typedef const uint32_t& const_reference;
//   typedef value_type* pointer;
//   typedef const value_type* const_pointer;
// };

// template <>
// struct DataTypeTraits<DataType_DT_UINT64> {
//   typedef uint64_t value_type;
//   typedef uint64_t& reference;
//   typedef const uint64_t& const_reference;
//   typedef value_type* pointer;
//   typedef const value_type* const_pointer;
// };

template <>
struct DataTypeTraits<DataType_DT_FLOAT> {
  typedef float value_type;
  typedef float& reference;
  typedef const float& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
};

}  // namespace tars