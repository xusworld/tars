#pragma once

#include <memory>

#include "tars/core/buffer.h"
#include "tars/core/macro.h"
#include "tars/core/tensor_shape.h"
#include "tars/core/type_traits.h"
#include "tars/core/types.h"
#include "tars/core/utils.h"
#include "tars/ir/types_generated.h"

namespace ace {

template <DataType data_type>
class Tensor {
 public:
  // C++ type traits trick
  typedef typename DataTypeTraits<data_type>::value_type T;
  typedef typename DataTypeTraits<data_type>::pointer TPtr;

  Tensor() {}

  Tensor(const RuntimeType rtype = RuntimeType::CPU,
         const TensorShape &shape = TensorShape(),
         const DataFormat dformat = DataFormat_NCHW,
         const std::string &name = "")
      : rtype_(rtype),
        dtype_(data_type),
        shape_(shape),
        valid_shape_(shape),
        dformat_(dformat),
        name_(name) {
    const int32_t elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
#ifdef ACE_DEBUG
    LOG(INFO) << "DataType: " << data_type << " , elements: " << elements;
#endif
    buff_ = std::make_shared<Buffer<T>>(rtype, data_type, elements);
  }

  Tensor(const Tensor &tensor) {
    shape_ = tensor.shape_;
    valid_shape_ = tensor.valid_shape_;
    offset_ = tensor.offset_;
    dtype_ = tensor.dtype_;
    dformat_ = tensor.dformat_;
    shared_ = tensor.shared_;
    buff_ = tensor.buff_;
  }

  Tensor(Tensor &&other) { *this = other; }

  Tensor &operator=(Tensor &&other) {
    shape_ = other.shape_;
    valid_shape_ = other.valid_shape_;
    offset_ = other.offset_;
    dtype_ = other.dtype_;
    dformat_ = other.dformat_;
    shared_ = other.shared_;
    buff_ = other.buff_;
    return *this;
  }

  Tensor &operator=(Tensor &other) {
    shape_ = other.shape_;
    valid_shape_ = other.valid_shape_;
    offset_ = other.offset_;
    dtype_ = other.dtype_;
    dformat_ = other.dformat_;
    shared_ = other.shared_;
    buff_ = other.buff_;
    return *this;
  }

  ~Tensor() {
    LOG(INFO) << "shared: " << shared_ << " , use_count: " << buff_.use_count();
    if (!shared_ && buff_.use_count() == 1) {
      // buff_.reset(nullptr);
    }
  }

  // returns true if the tensor is empty (or uninitialized)
  bool empty() const noexcept { return shape_.size() == 0; }

  // returns the total number of elements in the tensor
  int32_t size() const { return shape_.size(); }

  // returns the capacity of this tensor
  int32_t capacity() const { return buff_->capacity(); }

  // returns the total buffer bytes in the tensor
  int32_t bytes() const { return shape_.size() * DataType2Bytes(dtype_); }

  // returns the rank of the tensor
  int32_t rank() const { return shape_.dims(); };

  // returns the shape of the tensor
  TensorShape mutable_shape() const { return shape_; }
  const TensorShape shape() const { return shape_; };

  // return the data of the tensor
  const T *data() const {
    LOG(INFO) << "buff_.use_count(): " << buff_.use_count();
    LOG(INFO) << "buff_.get() " << buff_.get();
    return buff_.get()->data();
  }
  T *mutable_data() const { return buff_.get()->mutable_data(); }

  // returns the valid shape of the tensor
  TensorShape mutable_valid_shape() const { return valid_shape_; }
  const TensorShape valid_shape() const { return valid_shape_; };

  // returns the offset of the tensor
  TensorShape mutable_offset() const { return offset_; }
  const TensorShape offset() const { return offset_; };

  // returns the DataType of the tensor
  DataType dtype() const { return dtype_; }

  // returns the DataFormat of the tensor
  DataFormat dformat() const { return dformat_; }

  // returns the RuntimeType of the tensor
  RuntimeType rtype() const { return rtype_; }

  // reshape the tensor
  Status reshape(const TensorShape &shape) { return Status::OK(); }
  Status reshape(const std::vector<int32_t> &dims) {
    if (!shared_) {
      this->buff_.get()->resize(dims);
      this->shape_ = dims;
      this->valid_shape_ = dims;
    } else {
      LOG(INFO) << "a shard tensor, cannot resize, please check";
    }
    return Status::OK();
  }

  // reset memory
  Status reset(const T val) {
    LOG(INFO) << "reset val: " << val;
    this->buff_.get()->reset(val);

    return Status::OK();
  }

  Status squeeze() { return Status::OK(); }

  Status clear() {
    if (buff_.use_count() == 1) {
      // // release ownership
      buff_.reset();
      this->name_ = "";
      // z
      this->shape_ = TensorShape();
      this->valid_shape_ = TensorShape();
      this->offset_ = TensorShape();
    }
    return Status::OK();
  }

  // TODO
  Status set_dtype(const DataType dtype) {
    if (dtype_ != dtype) {
      dtype_ = dtype;
      const int32_t type_bytes = DataType2Bytes(dtype);

      if (buff_->capacity() < shape_.size() * type_bytes) {
        if (shared_) {
          LOG(FATAL) << "tensor is shared, memory can not be re-alloced";
          return Status::ERROR();
        }

        buff_->reallocate(shape_.size() * type_bytes);
      }
    }
    return Status::OK();
  }

 private:
  // tensor's name
  std::string name_;
  // tensor's runtime type
  RuntimeType rtype_;
  // tensor's data type
  DataType dtype_;
  // tensor's data format
  DataFormat dformat_;
  // tensor's shape
  TensorShape shape_;
  // Represent the mem you have right to access shape.
  TensorShape valid_shape_;
  // Represent the offset idx between _shape and _real_shape.
  TensorShape offset_;
  // tensor's buffer
  std::shared_ptr<Buffer<T>> buff_;

  bool shared_ = false;
};

}  // namespace ace
