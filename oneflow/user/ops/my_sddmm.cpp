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
  /*
  -输入tensor
    x：∈R(|V| x d1)，dense matrix
    y：∈R(|V| x d2)，dense matrix
    w：∈R(|ε| x d3)，dense matrix
    A(sparse matrix): 分解成3个tensor 即csr_row，csr_col，csr_data 
    */
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);  
  const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
  const user_op::TensorDesc& w = ctx->InputTensorDesc("w", 0);  
  const user_op::TensorDesc& csr_row = ctx->InputTensorDesc("csr_row", 0);
  const user_op::TensorDesc& csr_col = ctx->InputTensorDesc("csr_col", 0);
  const user_op::TensorDesc& csr_data = ctx->InputTensorDesc("csr_data", 0);
  
  //形状检查
  CHECK_GE_OR_RETURN(x.shape().NumAxes(), 2);
  CHECK_GE_OR_RETURN(y.shape().NumAxes(), 2);
  CHECK_GE_OR_RETURN(w.shape().NumAxes(), 2);
  int64_t num_edge = w.shape().At(1);  
  CHECK_GE_OR_RETURN(csr_col.shape().At(0), num_edge);
  CHECK_GE_OR_RETURN(csr_data.shape().At(0), num_edge);
  int64_t num_node = x.shape().At(1);  
  CHECK_GE_OR_RETURN(y.shape().At(1), num_node);
  CHECK_GE_OR_RETURN(csr_row.shape().At(0), num_node);

  /*
  -输出tensor
    z：∈R(|V| x d4)，dense matrix
    */
  ctx->SetOutputShape("z", 0, Shape({num_node}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MySDDMMOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

// Split, Broadcast, 定义分布式策略
/* static */ Maybe<void> MySDDMMOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("y", 0))
      .Broadcast(user_op::OpArg("w", 0))
      .Split(user_op::OpArg("z", 0), 0)
      .Broadcast(user_op::OpArg("csr_row", 0))
      .Broadcast(user_op::OpArg("csr_col", 0))
      .Broadcast(user_op::OpArg("csr_data", 0))
      .Build();
  return Maybe<void>::Ok();
}

//判断X，W输入数据类型是否一致，根据输入的类型推导输出的类型
/* static */ Maybe<void> MySDDMMOp::InferDataType(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("y", 0), dtype)
      << "Tensor Y datatype should be equal to Tensor X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("w", 0), dtype)
      << "Tensor W datatype should be equal to Tensor X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("csr_row", 0), dtype)
      << "Tensor csr_row datatype should be equal to Tensor X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("csr_col", 0), dtype)
      << "Tensor csr_col datatype should be equal to Tensor X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("csr_data", 0), dtype)
      << "Tensor csr_data datatype should be equal to Tensor X. ";
  ctx->SetOutputDType("z", 0, dtype);
  return Maybe<void>::Ok();
}



}  // namespace oneflow