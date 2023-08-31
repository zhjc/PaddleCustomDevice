
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
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "self_attention_kv_cache_fusion_ops_gpt3_runner_builder.h"
#include "self_attention_kv_cache_fusion_gpt3_operation.h"
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheFusionGPT3Operation::SelfAttentionKvCacheFusionGPT3Operation(const SelfAttentionKvCacheFusionParam &param)
    : Operation("SelfAttentionKvCacheFusionGPT3Operation"), param_(param)
{
    runnerBuilders_ = {new SelfAttentionKvCacheFusionOpsGPT3RunnerBuilder(param_)};
}

SelfAttentionKvCacheFusionGPT3Operation::~SelfAttentionKvCacheFusionGPT3Operation() {}

uint64_t SelfAttentionKvCacheFusionGPT3Operation::GetInTensorCount() const { return 7; }

uint64_t SelfAttentionKvCacheFusionGPT3Operation::GetOutTensorCount() const { return 1; }

AsdOps::Status
SelfAttentionKvCacheFusionGPT3Operation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.resize(GetOutTensorCount());
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1)); // batch == 1
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) / 3);
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionKvCacheFusionGPT3Operation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer