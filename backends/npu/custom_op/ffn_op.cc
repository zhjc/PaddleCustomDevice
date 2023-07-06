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

std::vector<paddle::Tensor> MyFfnOp(const paddle::Tensor& x,
                                     const paddle::Tensor& y) {
  // ONLY FOR PASS TEST
  return {paddle::add(x, y)};
}

std::vector<std::vector<int64_t>> MyFfnOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape) {
  // ONLY FOR PASS TEST        
  return {y_shape};
}

PD_BUILD_OP(Ffn_pass_test)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MyFfnOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MyFfnOpInferShape));  // neccessary if the op has muti_inputs
