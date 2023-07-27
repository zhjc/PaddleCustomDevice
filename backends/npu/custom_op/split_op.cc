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

#include "paddle/extension.h"

std::vector<paddle::Tensor> MySplitOp(const paddle::Tensor& x) {
  // ONLY FOR PASS TEST
  return {paddle::add(x, x), paddle::add(x, x), paddle::add(x, x)};
}

std::vector<std::vector<int64_t>> MySplitOpInferShape(
    const std::vector<int64_t>& x_shape) {
  // ONLY FOR PASS TEST        
  std::vector<int64_t> new_shape1;
  std::vector<int64_t> new_shape2;
  std::vector<int64_t> new_shape3;
  for (int i = 0;i < x_shape.size();i++) {
    if (i == x_shape.size() - 1) {
      new_shape1.push_back(x_shape[i] / 3);
      new_shape2.push_back(x_shape[i] / 3);
      new_shape3.push_back(x_shape[i] / 3);
    } else {
      new_shape1.push_back(x_shape[i]);
      new_shape2.push_back(x_shape[i]);
      new_shape3.push_back(x_shape[i]);
    }
  }
  return {new_shape1, new_shape2, new_shape3};
}

PD_BUILD_OP(split_pass_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"axis: int64_t"})
    .Attrs({"num: int64_t"})
    .SetKernelFn(PD_KERNEL(MySplitOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MySplitOpInferShape));  // neccessary if the op has muti_inputs
