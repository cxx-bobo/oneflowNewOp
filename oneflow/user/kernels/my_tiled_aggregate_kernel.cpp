/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

template<typename T>
class MyTiledAggregateKernel final : public user_op::OpKernel {
 public:
  MyTiledAggregateKernel() = default;
  ~MyTiledAggregateKernel() = default;

 private:
 //Compute必须重写，实现具体的运算逻辑
  void Compute(user_op::KernelComputeContext* ctx) const override {
    //传入Tensor4ArgNameAndIndex的字符串要和之前在OneFlowUserOps.td设置的名称一致
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    CHECK_EQ(x->shape_view().NumAxes(), 2) << "x Numdims should be equal to 2. ";
    const DataType data_type = x->data_type();
    CHECK_EQ(w->shape_view().NumAxes(), 2) << "w Numdims should be equal to 2. ";
    CHECK_EQ(x->data_type(), data_type) << "Matrix X Datatype should be equal to Vector b";
    CHECK_EQ(b->shape_view().NumAxes(), 1) << "b Numdims should be equal to 1. ";
    CHECK_EQ(b->data_type(), data_type) << "Matrix X Datatype should be equal to vector b";

    CHECK_EQ(y->shape_view().NumAxes(), 2) << "y Numdims should be equal to 2. ";
    CHECK_EQ(y->data_type(), data_type) << "y Datatype should be equal to input's. ";

    //cpu运算逻辑
    const T* x_ptr = x->dptr<T>();
    const T* w_ptr = w->dptr<T>();
    const T* b_ptr = b->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    // std::vector<int>* x_ptr = x->dptr();
    // std::vector<int>* w_ptr = w->dptr();
    // std::vector<int>* b_ptr = b->dptr();
    // std::vector<int>* y_ptr = y->mut_dptr();

    //矩阵size是N*N
    const int N = x->shape_view().At(0);
    //const int N = 1 << 10;

    // For every row...
    for (int i = 0; i < N; i++) {
    // For every column...
      for (int j = 0; j < N; j++) {
        // For every element in the row-column pair
        int y_index = i * N + j;
        for (int k = 0; k < N; k++) {
        // Accumulate the partial results
          y_ptr[y_index] += w_ptr[i * N + k] * x_ptr[k * N + j];
        }
        y_ptr[y_index] += b_ptr[j];
      }
    }
  }
  //AlwaysComputeWhenAllOutputsEmpty必须重写，若即使输出为空也需要调用kernel进行计算则返回true
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

//调用 REGISTER_USER_KERNEL 注册kernel类
#define REGISTER_MY_TILED_AGGREGATE_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("my_tiled_aggregate")                          \
      .SetCreateFn<MyTiledAggregateKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MY_TILED_AGGREGATE_KERNEL(float)
REGISTER_MY_TILED_AGGREGATE_KERNEL(double)
REGISTER_MY_TILED_AGGREGATE_KERNEL(uint8_t)
REGISTER_MY_TILED_AGGREGATE_KERNEL(int8_t)
REGISTER_MY_TILED_AGGREGATE_KERNEL(int32_t)
REGISTER_MY_TILED_AGGREGATE_KERNEL(int64_t)

}  // namespace oneflow