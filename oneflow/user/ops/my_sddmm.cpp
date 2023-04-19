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

// 验证输入张量形状，推导输出张量形状
/* static */ Maybe<void> MySDDMMOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  /*输入tensor
    x: 属于R(|V| x d1)，dense matrix
    y: 属于R(|V| x d2)，dense matrix
    w: 属于R(|V| x d3)，dense matrix
    A: sparse matrix 
    */
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);  
  const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
  const user_op::TensorDesc& w = ctx->InputTensorDesc("w", 0);  
  const user_op::TensorDesc& A = ctx->InputTensorDesc("A", 0);
  
  //w, x, y 都是二维
  CHECK_GE_OR_RETURN(x.shape().NumAxes(), 2);
  CHECK_GE_OR_RETURN(w.shape().NumAxes(), 2);
  CHECK_GE_OR_RETURN(y.shape().NumAxes(), 2);

  //A是csr_matrix
  // int v_num = A.col_indices().size(0);
  // Tensor e = A.crow_indices();
  // int e_num=e[csr.crow_indices().size(0)-1];

//   int64_t k = x.shape().At(1);  
//   //矩阵X的列数需等于向量b的行数
//   CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";
  
  /*输出tensor
    z: 属于R(|V| x d4)，dense matrix
    */
  Shape z_shape = x.shape();
  ctx->SetOutputShape("z", 0, z_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MySDDMMOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

// Split, Broadcast, 定义分布式策略
/* static */ Maybe<void> MySDDMMOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("w", 0))
      .Split(user_op::OpArg("x", 0), 1)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(user_op::OpArg("y", 0), 1)
      .Build();
  return Maybe<void>::Ok();
}

//判断X，W输入数据类型是否一致，根据输入的类型推导输出的类型
/* static */ Maybe<void> MySDDMMOp::InferDataType(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("w", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), dtype)
      << "Matrix W datatype should be equal to Matrix X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "Matrix W datatype should be equal to vector b. ";
  ctx->SetOutputDType("y", 0, dtype);
  return Maybe<void>::Ok();
}



}  // namespace oneflow