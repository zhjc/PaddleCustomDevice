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
#ifndef ACLTRANSFOERM_PARAMS_SELFATTENTION_KV_CACHE_GPT3_H
#define ACLTRANSFOERM_PARAMS_SELFATTENTION_KV_CACHE_GPT3_H
namespace AclTransformer {
struct SelfAttentionKvCacheGPT3Param {
    bool transKey = false;
    int64_t head_dim = 0;
    int64_t head_num = 0;
    int64_t layer_num = 0;
    std::string model = "gpt3";
};
} // namespace AclTransformer
#endif