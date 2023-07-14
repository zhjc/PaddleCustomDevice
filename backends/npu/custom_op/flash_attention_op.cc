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
#include "kernels/funcs/npu_op_runner.h"

std::vector<std::vector<int64_t>> FlashAttentionOpInferShape(
    const std::vector<int64_t>& query_shape) {
  std::vector<int64_t> output_shape = query_shape;
  return {output_shape};
}

std::vector<paddle::Tensor> FlashAttentionOp(const paddle::Tensor& query,
                                             const paddle::Tensor& cache_key,
                                             const paddle::Tensor& cache_value,
                                             const paddle::Tensor& batch,
                                             const paddle::Tensor& q_seqlen,
                                             const paddle::Tensor& kv_seqlen,
                                             const paddle::Tensor& attention_mask) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  auto cache_key_tensor = static_cast<const phi::DenseTensor*>(cache_key.impl().get());
  auto cache_value_tensor = static_cast<const phi::DenseTensor*>(cache_value.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor*>(attention_mask.impl().get());
  auto batch_tensor = static_cast<const phi::DenseTensor*>(batch.impl().get());
  auto q_seqlen_tensor = static_cast<const phi::DenseTensor*>(q_seqlen.impl().get());
  auto kv_seqlen_tensor = static_cast<const phi::DenseTensor*>(kv_seqlen.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  
  auto out_shape = FlashAttentionOpInferShape(query.shape()).at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), query_tensor->dtype());

  const auto& runner =
      NpuOpRunner("FlashAttentionV2", {*query_tensor, *cache_key_tensor, *cache_value_tensor, 
      *batch_tensor, *q_seqlen_tensor, *kv_seqlen_tensor, *attention_mask_tensor}, {*out_tensor}, {});
  runner.Run(stream);    
  
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(flash_attention)
    .Inputs({"Query", "CacheKey", "CacheValue", "Batch", "QSeqLen", "KVSeqLen", "AttentionMask"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(FlashAttentionOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        FlashAttentionOpInferShape));  // neccessary if the op has muti_inputs
