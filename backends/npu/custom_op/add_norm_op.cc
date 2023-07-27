// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> AddNormOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  return {x_shape};
}

std::vector<paddle::Tensor> AddNormOp(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::Tensor& weight,
    const paddle::Tensor& bias) {
  // 1. preprocessing: stream, input tensors;
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto weight_tensor = static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());
  
  // 2. create output tensor: tensor, out_shape, alloc;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
	  
  auto out_shape = AddNormOpInferShape(
      x.shape(), y.shape(), weight.shape(), bias.shape()).at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));
  
  dev_ctx->Alloc(out_tensor.get(), x_tensor->dtype());
  
  /*
  // 3. run;
  use this
  const auto& add_runner =
      NpuOpRunner("AddNorm", {*x_tensor, *y_tensor, *weight_tensor, *bias_tensor}, {*out_tensor}, {});
  add_runner.Run(stream);

  
  // 4. out_tensor->paddle::Tensor.
  return {paddle::Tensor(out_tensor)};
  */
  const auto& runner =
      NpuOpRunner("Add", {*x_tensor, *y_tensor}, {*out_tensor}, {});
  runner.Run(stream);
  // auto add_res_tensor = std::make_shared<phi::DenseTensor>();
  // add_res_tensor->ShareDataWith(out_tensor);
  //paddle::Tensor add_res(out_tensor);
  //paddle::Tensor output = paddle::nn:functional::layer_norm(add_res, 0, weight=weight, bias=bias);

  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(add_norm)
    .Inputs({"X", "Y", "Weight", "Bias"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddNormOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AddNormOpInferShape));  // neccessary if the op has muti_inputs
