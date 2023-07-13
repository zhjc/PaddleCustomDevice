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

std::vector<paddle::Tensor> MultiheadAttentionOp(const paddle::Tensor& q,
                                            const paddle::Tensor& concated_k,
                                            const paddle::Tensor& concated_v,
                                            const paddle::Tensor& attn_mask) {
  std::cout << "Run in MultiheadAttentionOp" << std::endl;
  return {paddle::add(q, paddle::add(concated_k, concated_v))};
}

std::vector<std::vector<int64_t>> MultiheadAttentionOpInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& concated_k_shape,
    const std::vector<int64_t>& concated_v_shape,
    const std::vector<int64_t>& attn_mask_shape) {
  return {q_shape};
}


PD_BUILD_OP(multihead_attention_op)
  .Inputs({"Q", "ConcatedK", "ConcatedV", "AttnMask"})
  .Outputs({"Out"})
  .SetKernelFn(PD_KERNEL(MultiheadAttentionOp))
  .SetInferShapeFn(PD_INFER_SHAPE(
    MultiheadAttentionOpInferShape
  ));