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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

//namespace {

template<typename T>
__global__ void TiledAggregateGpu(
    const T *matrix_W, 
    const T *matrix_H, 
    const T *vector_b,
    const int tile_size,
    T *c,
    const int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  extern __shared__ int tile[];
  int* tile_W = tile;
  int* tile_H = tile+tile_size*tile_size;

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    tile_W[threadIdx.y * blockDim.x + threadIdx.x] = matrix_W[row * N + i + threadIdx.x];
    tile_H[threadIdx.y * blockDim.x + threadIdx.x] = matrix_H[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          tile_W[threadIdx.y * blockDim.x + j] * tile_H[j * blockDim.x + threadIdx.x];
    }
    
    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  tmp += vector_b[col];
  c[row * N + col] = tmp;
}

//}  // namespace

template<typename T>
class GpuMyTiledAggregateKernel final : public user_op::OpKernel {
 public:
  GpuMyTiledAggregateKernel() = default;
  ~GpuMyTiledAggregateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
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

    //int N = x.shape().At(0);  //x，w，y矩阵的size都是N*N
    int N = 1 << 10;
    int threads_num = N*N;  //计算要启动的线程数量
    // Threads per CTA dimension
    int THREADS = 32;
    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;
    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    // obtain shared memory size for each thread block(tile_A+tile_B,所以乘2)
    int shared_memory_size = 2*THREADS*THREADS*sizeof(int);
    
    // RUN_CUDA_KERNEL((TiledAggregateGpu<T1,T2>), ctx->stream(), threads_num, 
    //                 w->dptr<T1>(), x->dptr<T1>(), b->dptr<T2>(), THREADS, y->mut_dptr<T1>(), N);
    TiledAggregateGpu<T><<<blocks, threads, shared_memory_size>>>
                    (w->dptr<T>(), x->dptr<T>(), b->dptr<T>(), THREADS, y->mut_dptr<T>(), N);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(dtype)                  \
  REGISTER_USER_KERNEL("my_tiled_aggregate")                           \
      .SetCreateFn<GpuMyTiledAggregateKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(float)
REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(double)
REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(uint8_t)
REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(int8_t)
REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(int32_t)
REGISTER_GPU_MY_TILED_AGGREGATE_KERNEL(int64_t)

}  // namespace oneflow