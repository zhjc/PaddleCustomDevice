/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#ifndef SELFATTENTION_WITHOUT_KVCACHE_OPS_RUNNER_BUILDER_H
#define SELFATTENTION_WITHOUT_KVCACHE_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include "acltransformer/runner_builder.h"
#include "self_attention_kv_cache_gpt3.h"
#include "self_attention_without_kv_cache_gpt3_ops_runner.h"

namespace AclTransformer {
class SelfAttentionWithoutKvCacheGPT3OpsRunnerBuilder : public RunnerBuilder {
 public:
  SelfAttentionWithoutKvCacheGPT3OpsRunnerBuilder(
      const SelfAttentionKvCacheGPT3Param &param)
      : param_(param) {}
  virtual ~SelfAttentionWithoutKvCacheGPT3OpsRunnerBuilder() = default;
  Runner *Build() override {
    if (param_.model == "gpt3") {
      return new SelfAttentionWithoutKvCacheOpsGPT3Runner(param_);
    } else {
      ASD_LOG(ERROR) << "invalid param_.model:" << param_.model;
      return nullptr;
    }
  }

 private:
  SelfAttentionKvCacheGPT3Param param_;
};

}  // namespace AclTransformer
#endif