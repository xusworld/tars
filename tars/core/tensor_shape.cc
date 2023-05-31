#include <numeric>

#include "tars/core/tensor_shape.h"

namespace tars {

TensorShape& TensorShape::operator=(const TensorShape& right) {
  this->assign(right.begin(), right.end());
  return *this;
}

// TensorShape& TensorShape::operator=(const std::vector<int>& right) {
//   this->assign(right.begin(), right.end());
//   return *this;
// }

TensorShape TensorShape::operator+(const TensorShape& shape) {
  TensorShape tmp_shape(*this);
  int* p = data();

  for (size_t i = 0; i < size(); i++) {
    tmp_shape[i] = p[i] + shape[i];
  }

  return tmp_shape;
}

TensorShape TensorShape::operator-(const TensorShape& shape) {
  TensorShape tmp_shape(*this);
  int* p = data();

  for (size_t i = 0; i < size(); i++) {
    tmp_shape[i] = p[i] - shape[i];
  }

  return tmp_shape;
}

bool TensorShape::operator<(const TensorShape& shape) const {
  const auto dims = this->size();

  if (dims != shape.size()) return false;

  for (size_t i = 0; i < dims; i++) {
    if (this->at(i) != shape[i]) return false;
  }

  return true;
}

bool TensorShape::operator<=(const TensorShape& shape) const {
  const auto dims = this->size();

  if (dims != shape.size()) return false;

  const int* p = data();
  for (size_t i = 0; i < dims; i++) {
    if (this->at(i) <= shape[i]) return false;
  }

  return true;
}

bool TensorShape::operator>(const TensorShape& shape) const {
  const auto dims = this->size();

  if (dims != shape.size()) return false;

  const int* p = data();
  for (size_t i = 0; i < dims; i++) {
    if (this->at(i) > shape[i]) return false;
  }

  return true;
}

bool TensorShape::operator>=(const TensorShape& shape) const {
  const auto dims = this->size();

  if (dims != shape.size()) return false;

  const int* p = data();
  for (size_t i = 0; i < dims; i++) {
    if (this->at(i) >= shape[i]) return false;
  }

  return true;
}

bool TensorShape::operator==(const TensorShape& shape) const {
  const auto dims = this->size();

  if (dims != shape.size()) return false;

  const int* p = data();
  for (size_t i = 0; i < dims; i++) {
    if (this->at(i) == shape[i]) return false;
  }

  return true;
}

TensorShape TensorShape::zero(const TensorShape& right) {
  TensorShape sh = right;

  for (int i = 0; i < right.size(); ++i) {
    sh[i] = 0;
  }

  return sh;
}

TensorShape TensorShape::minusone(const TensorShape& right) {
  TensorShape sh = right;

  for (int i = 0; i < right.size(); ++i) {
    sh[i] = -1;
  }

  return sh;
}

std::string TensorShape::debug_string() const {
  if (this->size() == 0) {
    LOG(INFO) << "A uninitialized tensor shape, please check.";
    return "{}";
  }

  const std::string prefix = "{";
  const std::string suffix = "}";
  const std::string separator = ",";
  const auto len = this->size();

  std::stringstream ss;
  ss << prefix;
  for (auto idx = 0; idx < len - 1; ++idx) {
    ss << this->at(idx) << separator;
  }

  ss << this->at(len - 1) << suffix << ", dims: " << this->dims();
  return ss.str();
}

}  // namespace tars