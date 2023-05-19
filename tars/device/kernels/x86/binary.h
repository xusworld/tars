#pragma once

#include <iostream>
#include <vector>

namespace ace {
namespace device {
namespace x86 {

template <typename T>
void _naive_binary_add(T* outputs, const T* lhs, const T* rhs,
                       const int32_t size);

template <typename T>
void _naive_binary_mul(T* outputs, const T* lhs, const T* rhs,
                       const int32_t size);

template <typename T>
void _naive_binary_min(T* outputs, const T* lhs, const T* rhs,
                       const int32_t size);

template <typename T>
void _naive_binary_max(T* outputs, const T* lhs, const T* rhs,
                       const int32_t size);

template <typename T>
void _naive_binary_mean(T* outputs, const T* lhs, const T* rhs,
                        const int32_t size);

template <typename T>
void _naive_binary_div(T* outputs, const T* lhs, const T* rhs,
                       const int32_t size);

template <typename T>
void _naive_binary_greater_equal(bool* outputs, const T* lhs, const T* rhs,
                                 const int32_t size);

template <typename T>
void _naive_binary_greater_than(bool* outputs, const T* lhs, const T* rhs,
                                const int32_t size);

template <typename T>
void _naive_binary_less_equal(bool* outputs, const T* lhs, const T* rhs,
                              const int32_t size);

template <typename T>
void _naive_binary_less_than(bool* outputs, const T* lhs, const T* rhs,
                             const int32_t size);

template <typename T>
void _naive_binary_equal_to(bool* outputs, const T* lhs, const T* rhs,
                            const int32_t size);

template <typename T>
void _naive_binary_not_equal(bool* outputs, const T* lhs, const T* rhs,
                             const int32_t size);

}  // namespace x86
}  // namespace device
}  // namespace ace