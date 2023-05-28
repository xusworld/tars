#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "tars/core/utils.h"
#include "tars/ir/types_generated.h"

namespace tars {

namespace {

using Vector = std::vector<int32_t>;

}  // namespace

class TensorShape : public Vector {
 public:
  TensorShape() {}

  TensorShape(const std::vector<int32_t>& dims) {
    this->assign(dims.begin(), dims.end());
  }

  TensorShape(const TensorShape& shape) {
    this->assign(shape.begin(), shape.end());
  }

  TensorShape(TensorShape&& shape) { this->assign(shape.begin(), shape.end()); }

  ~TensorShape() = default;

  // Basic operator overloading function.
  TensorShape& operator=(const TensorShape& shape);
  TensorShape operator+(const TensorShape& shape);
  TensorShape operator-(const TensorShape& shape);

  int32_t& operator[](const int i) {
    if (i >= this->size()) {
      this->resize(i + 1);
    }
    return this->at(i);
  }

  const int32_t& operator[](const int i) const { return this->at(i); }

  bool operator<(const TensorShape& shape) const;
  bool operator<=(const TensorShape& shape) const;
  bool operator>(const TensorShape& shape) const;
  bool operator>=(const TensorShape& shape) const;
  bool operator==(const TensorShape& shape) const;

  // Return a sub shape.
  TensorShape subshape(const int32_t start, const int32_t end) const {
    CHECK_LT(start, end)
        << "TensorShape::elems 'start' value should less than 'end'";
    CHECK_LE(start, this->size())
        << "TensorShape::elems 'start' value should less than this->size()";
    TensorShape shape;
    for (auto idx = start; idx < end; ++idx) {
      shape.push_back(this->at(idx));
    }
    return shape;
  }

  // If a empty tensor shape.
  bool empty() const { return this->size() == 0; }

  // Return number of dimensions in this shape.
  int32_t dims() const { return this->size(); }

  // Reset the value of tensor shape.
  void reset(const std::vector<int32_t>& dims) {
    return this->assign(dims.begin(), dims.end());
  }

  // Reset the value of tensor shape.
  void reset(const TensorShape& shape) {
    this->assign(shape.begin(), shape.end());
  }

  // Return tensor's element size.
  int32_t elems() const {
    return std::accumulate(this->begin(), this->end(), 1,
                           std::multiplies<int32_t>());
  }

  // Return tensor's element size.
  int32_t elems(const int32_t start, const int32_t end) const {
    CHECK_LT(start, end)
        << "TensorShape::elems 'start' value should less than 'end'";
    CHECK_LE(start, this->size())
        << "TensorShape::elems 'start' value should less than this->size()";

    return std::accumulate(this->begin() + start, this->begin() + end, 1,
                           std::multiplies<int32_t>());
  }

  // Append a value.
  void append(const int32_t val) { this->push_back(val); }

  // A helper function, only use for debug mode.
  std::string debug_string() const;

  // Return a zero tensor shape has the same dimension of input shape.
  static TensorShape zero(const TensorShape& shape);

  // Return a minus one tensor shape has the same dimension of input shape.
  static TensorShape minusone(const TensorShape& shape);
};

inline std::ostream& operator<<(std::ostream& os, TensorShape& shape) {
  os << shape.debug_string();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << shape.debug_string();
  return os;
}

}  // namespace tars
