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
#ifndef ACLTRANSFORMER_SELFATTENTION_WITHOUT_KV_CACHE_GPT3_OPERATION_H
#define ACLTRANSFORMER_SELFATTENTION_WITHOUT_KV_CACHE_GPT3_OPERATION_H
#include "acltransformer/operation.h"
#include "self_attention_kv_cache_gpt3.h"

namespace AclTransformer {
class SelfAttentionWithoutKvCacheGPT3Operation : public Operation {
 public:
  SelfAttentionWithoutKvCacheGPT3Operation(const SelfAttentionKvCacheGPT3Param &param);
  ~SelfAttentionWithoutKvCacheGPT3Operation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
    AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;
    RunnerBuilder *FindBestRunnerBuilder() const override;

 private:
    SelfAttentionKvCacheGPT3Param param_;
};
}
#endif