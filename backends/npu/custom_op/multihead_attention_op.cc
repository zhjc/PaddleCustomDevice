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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "kernels/funcs/format_utils.h"
#include "acltransformer/params/self_attention_kv_cache_gpt3.h"
#endif

std::vector<std::vector<int64_t>> MultiheadAttentionOpInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& concated_k_shape,
    const std::vector<int64_t>& concated_v_shape,
    const std::vector<int64_t>& attn_mask_shape) {
  std::vector<int64_t> output_shape;
  output_shape.push_back(q_shape.at(0));
  output_shape.push_back(q_shape.at(1));
  output_shape.push_back(q_shape.at(2) * q_shape.at(3));
  return {output_shape};
}

std::vector<paddle::Tensor> MultiheadAttentionOp(const paddle::Tensor& query,
                                            const paddle::Tensor& concated_k,
                                            const paddle::Tensor& concated_v,
                                            const paddle::Tensor& attn_mask,
                                            float layer_num = 1.0) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  auto concated_k_tensor = static_cast<const phi::DenseTensor*>(
      concated_k.impl().get());
  auto concated_v_tensor = static_cast<const phi::DenseTensor*>(
      concated_v.impl().get());
  auto attn_mask_tensor = static_cast<const phi::DenseTensor*>(
      attn_mask.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  auto out_shape = MultiheadAttentionOpInferShape(query.shape(),
      concated_k.shape(), concated_v.shape(), attn_mask.shape()).at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));

#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
  dev_ctx->Alloc(out_tensor.get(), query_tensor->dtype());
  /* [bs, seq_len, head_num, head_dim] */
  int64_t head_dim = query.shape().at(3);
  int64_t head_num = query.shape().at(2);
  auto query_tensor_asd = ConvertDenseTensorToAsdTensor(*query_tensor);
  auto concated_k_tensor_asd =
      ConvertDenseTensorToAsdTensor(*concated_k_tensor);
  auto concated_v_tensor_asd =
      ConvertDenseTensorToAsdTensor(*concated_v_tensor);
  auto attn_mask_tensor_asd = ConvertDenseTensorToAsdTensor(*attn_mask_tensor);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out_tensor);

  AclTransformer::SelfAttentionKvCacheGPT3Param opParam = {false,
      head_dim, head_num, (int64_t)layer_num};
  AclTransformer::OperationCall opCall("SelfAttentionKvCacheGPT3Operation",
      opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {query_tensor_asd,
      concated_k_tensor_asd, concated_v_tensor_asd, attn_mask_tensor_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "Linear run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out_tensor)};
#else
  return {paddle::Tensor(out_tensor)};
#endif
}

PD_BUILD_OP(multihead_attention)
  .Inputs({"Q", "ConcatedK", "ConcatedV", "AttnMask"})
  .Outputs({"Out"})
  .Attrs({"layer_num: float"})
  .SetKernelFn(PD_KERNEL(MultiheadAttentionOp))
  .SetInferShapeFn(PD_INFER_SHAPE(
    MultiheadAttentionOpInferShape
  ));