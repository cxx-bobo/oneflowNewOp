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
/* static */ Maybe<void> MyTiledAggregateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  //Y=W*X+b, W是m*n维，X是n*p维，b是p*1维
  const user_op::TensorDesc& w = ctx->InputTensorDesc("w", 0);  
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);  
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t k = x.shape().At(1);  
  //矩阵X的列数需等于向量b的行数
  CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";
  //矩阵W和矩阵X是相同尺寸的
  CHECK_EQ_OR_RETURN(w.shape().NumAxes(), x.shape().NumAxes());
  CHECK_GE_OR_RETURN(x.shape().NumAxes(), 2);
  CHECK_GE_OR_RETURN(w.shape().At(0), x.shape().At(0));
  CHECK_GE_OR_RETURN(w.shape().At(1), x.shape().At(1));
  //输出矩阵形状
  Shape y_shape = x.shape();
  ctx->SetOutputShape("y", 0, y_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MyTiledAggregateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

// Split, Broadcast, 定义分布式策略
/* static */ Maybe<void> MyTiledAggregateOp::GetSbp(user_op::SbpContext* ctx) {
  //split(0)横着切，split(1)竖着切
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("w", 0))
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(user_op::OpArg("y", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

//判断X，W输入数据类型是否一致，根据输入的类型推导输出的类型
/* static */ Maybe<void> MyTiledAggregateOp::InferDataType(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("w", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("x", 0), dtype)
      << "Matrix W datatype should be equal to Matrix X. ";
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "Matrix W datatype should be equal to vector b. ";
  ctx->SetOutputDType("y", 0, dtype);
  return Maybe<void>::Ok();
}

// REGISTER_USER_OP_GRAD("my_tiled_aggregate")
//   .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
//     const std::string my_tiled_aggregate_grad_op_name = ctx->FwOp().op_name() + "_grad";
//     ctx->DefineOp(my_tiled_aggregate_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
//       return builder.OpTypeName("my_tiled_aggregate_grad")
//           .InputBind("dy", ctx->FwOp().output_grad("y", 0))
//           .InputBind("x", ctx->FwOp().input("x", 0))
//           // .Attr("alpha", ctx->FwOp().attr<float>("alpha"))
//           .Output("dx")
//           .Build();
//     });
//       ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
//                                 [&ctx, &my_tiled_aggregate_grad_op_name]() -> const std::string& {
//                                   return ctx->GetOp(my_tiled_aggregate_grad_op_name).output("dx", 0);
//                                 });
//       return Maybe<void>::Ok();
//     });

}  // namespace oneflow
 