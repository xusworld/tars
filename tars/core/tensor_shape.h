#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "tars/core/utils.h"
#include "tars/ir/types_generated.h"

namespace tars {

namespace {

constexpr int32_t kMaxTensorDims = 6;

using Vector = std::vector<int32_t>;

}  // namespace

class TensorShape : public Vector {
 public:
  TensorShape() {
    DLOG(INFO) << ", returns a default shape object.";
    // this->resize(kMaxTensorDims, 0);
  }

  TensorShape(const std::vector<int32_t>& dims) {
    this->assign(dims.begin(), dims.end());
  }

  TensorShape(const TensorShape& shape) {
    this->assign(shape.begin(), shape.end());
  }

  TensorShape(TensorShape&& shape) { this->assign(shape.begin(), shape.end()); }

  ~TensorShape() = default;

  // some operator overloading methods
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

  // returns a sub shape
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

  // is a empty tensor shape or not.
  bool empty() const { return this->size() == 0; }

  // returns the number of dimension of the shape.
  int32_t dims() const { return this->size(); }

  // returns the number of dimension of the shape.
  int32_t rank() const { return dims(); }

  // reset the value of the shape.
  void reset(const std::vector<int32_t>& dims) {
    return this->assign(dims.begin(), dims.end());
  }

  // reset the value of the shape.
  void reset(const TensorShape& shape) {
    this->assign(shape.begin(), shape.end());
  }

  // returns tensor's element size.
  int32_t elems() const {
    return std::accumulate(this->begin(), this->end(), 1,
                           std::multiplies<int32_t>());
  }

  // returns tensor's element size.
  int32_t elems(const int32_t start, const int32_t end) const {
    CHECK_LT(start, end)
        << "TensorShape::elems 'start' value should less than 'end'";
    CHECK_LE(start, this->size())
        << "TensorShape::elems 'start' value should less than this->size()";

    return std::accumulate(this->begin() + start, this->begin() + end, 1,
                           std::multiplies<int32_t>());
  }

  // append a value to the shape.
  void append(const int32_t val) { this->push_back(val); }

  // a helper function, only use for debug mode.
  std::string debug_string() const;

  // returns a zero tensor shape has the same dimension of input shape.
  static TensorShape zero(const TensorShape& shape);

  // returns a minus one tensor shape has the same dimension of input shape.
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
