/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "self_attention_without_kv_cache_gpt3_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "self_attention_without_kv_cache_gpt3_ops_runner_builder.h"

namespace AclTransformer {
  SelfAttentionWithoutKvCacheGPT3Operation::SelfAttentionWithoutKvCacheGPT3Operation(const SelfAttentionKvCacheGPT3Param &param)
      : Operation("SelfAttentionWithoutKvCacheGPT3Operation"), param_(param)
  {
    runnerBuilders_ = {new SelfAttentionWithoutKvCacheGPT3OpsRunnerBuilder(param_)};
  }

  SelfAttentionWithoutKvCacheGPT3Operation::~SelfAttentionWithoutKvCacheGPT3Operation() {}

  uint64_t SelfAttentionWithoutKvCacheGPT3Operation::GetInTensorCount() const { return 2; }

  uint64_t SelfAttentionWithoutKvCacheGPT3Operation::GetOutTensorCount() const { return 3; }

  AsdOps::Status SelfAttentionWithoutKvCacheGPT3Operation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
  {
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
    outTensorDescs.at(0).dims.push_back(param_.head_num * param_.head_dim);

    outTensorDescs.at(1) = inTensors.at(0).desc;
    outTensorDescs.at(1).dims.at(2) = param_.head_num;
    outTensorDescs.at(1).dims.at(3) = param_.head_dim;

    outTensorDescs.at(2) = inTensors.at(0).desc;
    outTensorDescs.at(2).dims.at(2) = param_.head_num;
    outTensorDescs.at(2).dims.at(3) = param_.head_dim;
    return AsdOps::Status::OkStatus();
  }

  RunnerBuilder *SelfAttentionWithoutKvCacheGPT3Operation::FindBestRunnerBuilder() const
  {
    size_t index = 0;

    return runnerBuilders_.at(index);
  }
} // namespace AclTransformer
