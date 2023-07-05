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

std::vector<std::vector<int64_t>> LinearOpInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  std::vector<int64_t> out_shape = input_shape;
  out_shape[out_shape.size() - 1] = weight_shape[weight_shape.size() - 1];
  return {out_shape};
}

std::vector<paddle::Tensor> LinearOp(
    const paddle::Tensor& input,
    const paddle::Tensor& weight,
    const paddle::Tensor& bias) {
  // 1. preprocessing: stream, input tensors;
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto input_tensor = static_cast<const phi::DenseTensor*>(input.impl().get());
  auto weight_tensor = static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());
  
  // 2. create output tensor: tensor, out_shape, alloc;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
	  
  auto out_shape = LinearOpInferShape(
      input.shape(), weight.shape(), bias.shape()).at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));
  
  dev_ctx->Alloc(out_tensor.get(), input_tensor->dtype());
  
  /*
  // 3. run;
  use this
  const auto& add_runner =
      NpuOpRunner("Linear", {*input_tensor, *weight_tensor, *bias_tensor}, {*out_tensor}, {});
  add_runner.Run(stream);

  
  // 4. out_tensor->paddle::Tensor.
  return {paddle::Tensor(out_tensor)};
  */
  paddle::Tensor mm_res = paddle::matmul(input, weight);
  auto mm_tensor = static_cast<const phi::DenseTensor*>(mm_res.impl().get());
  const auto& runner =
      NpuOpRunner("Add", {*mm_tensor, *bias_tensor}, {*out_tensor}, {});
  runner.Run(stream);
  
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(linear)
    .Inputs({"Input", "Weight", "Bias"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LinearOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LinearOpInferShape));  // neccessary if the op has muti_inputs
