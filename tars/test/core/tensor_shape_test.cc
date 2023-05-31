#include <gtest/gtest.h>

#include "tars/core/tensor_shape.h"

TEST(tensorshape, basic) {
  {
    tars::TensorShape shape;
    ASSERT_EQ(shape.dims(), 0);
    ASSERT_EQ(shape.empty(), true);
  }

  {
    tars::TensorShape shape({1, 224, 224, 3});
    ASSERT_EQ(shape.dims(), 4);
    ASSERT_EQ(shape.empty(), false);
    int elems = 1 * 224 * 224 * 3;
    ASSERT_EQ(shape.elems(), elems);
  }

  {
    tars::TensorShape shape = std::vector<int>({1, 224, 224, 3});
    ASSERT_EQ(shape.dims(), 4);
    ASSERT_EQ(shape.empty(), false);
    int elems = 1 * 224 * 224 * 3;
    ASSERT_EQ(shape.elems(), elems);
  }
}