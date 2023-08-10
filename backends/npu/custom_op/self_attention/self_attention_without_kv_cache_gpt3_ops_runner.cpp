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
#include "self_attention_without_kv_cache_gpt3_ops_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionWithoutKvCacheOpsGPT3Runner::SelfAttentionWithoutKvCacheOpsGPT3Runner(
    const SelfAttentionKvCacheGPT3Param &param) : OpsRunner("SelfAttentionWithoutKvCacheOpsGPT3Runner",
    RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param) {
  ASD_LOG(INFO) << "SelfAttentionWithoutKvCacheOpsGPT3Runner::"
                   "SelfAttentionWithoutKvCacheOpsGPT3Runner called"
                << "transKey: " << param_.transKey
                << ",head_dim: " << param_.head_dim
                << ",headNum: " << param_.head_num
                << ",layerNum: " << param_.layer_num;
  kernelGraph_.inTensors.resize(2);
  AsdOps::Tensor &mixedLayer = kernelGraph_.inTensors.at(0);
  AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(1);

  kernelGraph_.outTensors.resize(3);
  AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);
  AsdOps::Tensor &mixedKey = kernelGraph_.outTensors.at(1);
  AsdOps::Tensor &mixedValue = kernelGraph_.outTensors.at(2);

  kernelGraph_.internalTensors.resize(10);
  AsdOps::Tensor &mixedQuery = kernelGraph_.internalTensors.at(0);
  AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(1);
  AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(2);
  AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(3);
  AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(4);
  AsdOps::Tensor &scaleOut = kernelGraph_.internalTensors.at(5);
  AsdOps::Tensor &addOut = kernelGraph_.internalTensors.at(6);
  AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(7);
  AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(8);
  AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(9);

  kernelGraph_.nodes.resize(11);
  auto &splitNode = kernelGraph_.nodes.at(0);
  auto &permuteQNode = kernelGraph_.nodes.at(1);
  auto &mulsQNode = kernelGraph_.nodes.at(2);
  auto &permuteKNode = kernelGraph_.nodes.at(3);
  auto &bmmQkNode = kernelGraph_.nodes.at(4);
  auto &scaleNode = kernelGraph_.nodes.at(5);
  auto &addNode = kernelGraph_.nodes.at(6);
  auto &softMaxNode = kernelGraph_.nodes.at(7);
  auto &permuteVNode = kernelGraph_.nodes.at(8);
  auto &bmmVNode = kernelGraph_.nodes.at(9);
  auto &permuteContextNode = kernelGraph_.nodes.at(10);

  /* [bs, seq_len, 3 * hidden_size] -> [bs, seq_len, num_head, 3 * head_dim]
   * ->split  */
  splitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{3, 3}};
  splitNode.inTensors = {&mixedLayer};
  splitNode.outTensors = {&mixedQuery, &mixedKey, &mixedValue};
  splitNode.inTensorViewFuncs.resize(splitNode.inTensors.size());
  splitNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                       AsdOps::SVector<int64_t> &newDims) {
    newDims = {
        oldDims.at(0), oldDims.at(1), param_.head_num, 3 * param_.head_dim};
  };
  splitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
    runInfo.SetOpDesc(
        {0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  AsdOps::OpParam::Transpose permuteQNodeParam = {
      AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
  permuteQNode.opDesc = {0, "TransposeOperation", permuteQNodeParam};
  permuteQNode.inTensors = {&mixedQuery};
  permuteQNode.outTensors = {&transposedQ};

  /* [bs, num_head, seq_len, head_dim] */
  float varAttr = 1.0 / (sqrt(param_.head_dim) * (param_.layer_num));
  mulsQNode.opDesc = {0,
                      "ElewiseOperation",
                      AsdOps::OpParam::Elewise(
                          {AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
  mulsQNode.inTensors = {&transposedQ};
  mulsQNode.outTensors = {&divOut};
  mulsQNode.inTensorViewFuncs.resize(mulsQNode.inTensors.size());
  mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /* prepare k */
  /* k:[bs, seq_len_k, num_head, head_dim] -> [bs, num_head, seq_len_k,
   * head_dim] */
  AsdOps::OpParam::Transpose permuteKNodeParam = {
      AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
  permuteKNode.opDesc = {0, "TransposeOperation", permuteKNodeParam};
  permuteKNode.inTensors = {&mixedKey};
  permuteKNode.outTensors = {&transposedK};
  permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
  permuteKNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /*
   * q: [bs, num_head, seq_len_q, head_dim] -> [bs * num_head, seq_len_q,
   * head_dim] k: [bs, num_head, seq_len_k, head_dim] -> [bs * num_head,
   * seq_len_k, head_dim] transpose=ture: [bs * num_head, seq_len_k, head_dim]
   * -> [bs * num_head, head_dim, seq_len_k] out: [bs * num_head, seq_len_q,
   * seq_len_k]
   */
  bmmQkNode.opDesc = {0,
                      "MatMulOperation",
                      AsdOps::OpParam::MatMul({false, true, {/*oriShape*/}})};
  bmmQkNode.inTensors = {&divOut, &transposedK};
  bmmQkNode.outTensors = {&bmmQkOut};
  bmmQkNode.inTensorViewFuncs.resize(bmmQkNode.inTensors.size());
  bmmQkNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                      AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
  };
  bmmQkNode.inTensorViewFuncs[1] = [](const AsdOps::SVector<int64_t> &oldDims,
                                      AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
  };
  bmmQkNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /* [bs * num_head, seq_len_q, seq_len_k] -> [bs, num_head, seq_len_q,
   * seq_len_k] */
  float scale = param_.layer_num;
  scaleNode.opDesc = {0,
                      "ElewiseOperation",
                      AsdOps::OpParam::Elewise(
                          {AsdOps::OpParam::Elewise::ELEWISE_MULS, scale})};
  scaleNode.inTensors = {&bmmQkOut};
  scaleNode.outTensors = {&scaleOut};
  scaleNode.inTensorViewFuncs.resize(scaleNode.inTensors.size());
  scaleNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                       AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0) / param_.head_num,
               param_.head_num,
               oldDims.at(1),
               oldDims.at(2)};
  };
  scaleNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /* attention_mask:[bs, seq_len_q] -> [bs, 1, seq_len_q, 1] ->boradcast: */
  addNode.opDesc = {
      0,
      "BroadcastOperation",
      AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
  addNode.inTensors = {&scaleOut, &attention_mask};
  addNode.outTensors = {&addOut};
  addNode.inTensorViewFuncs.resize(addNode.inTensors.size());
  addNode.inTensorViewFuncs[1] = [](const AsdOps::SVector<int64_t> &oldDims,
                                    AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0), 1, oldDims.at(1), 1};
  };

  softMaxNode.opDesc = {
      0,
      "NormOperation",
      AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
  softMaxNode.inTensors = {&addOut};
  softMaxNode.outTensors = {&attentionProbs};

  /* prepare v */
  /* [bs, seq_len, num_head, head_dim] -> */
  AsdOps::OpParam::Transpose permuteVNodeParam = {
      AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
  permuteVNode.opDesc = {0, "TransposeOperation", permuteVNodeParam};
  permuteVNode.inTensors = {&mixedValue};
  permuteVNode.outTensors = {&transposedV};
  permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
  permuteVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /*
   * att: [bs, num_head, seq_len_q, seq_len_k] -> [bs * num_head, seq_len_q,
   * seq_len_k] v:   [bs, num_head, seq_len_v, head_dim]  -> [bs * num_head,
   * seq_len_v, head_dim] out: [bs * num_head, seq_len_q, head_dim]
   */
  bmmVNode.opDesc = {0,
                     "MatMulOperation",
                     AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
  bmmVNode.inTensors = {&attentionProbs, &transposedV};
  bmmVNode.outTensors = {&bmmVout};
  bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
  bmmVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                     AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
  };
  bmmVNode.inTensorViewFuncs[1] = [](const AsdOps::SVector<int64_t> &oldDims,
                                     AsdOps::SVector<int64_t> &newDims) {
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
  };
  bmmVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
    for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
      runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  };

  /* out: [bs, num_head, seq_len_q, head_dim]*/
  AsdOps::OpParam::Transpose permuteContextNodeParam = {
      AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
  permuteContextNode.opDesc = {
      0, "TransposeOperation", permuteContextNodeParam};
  permuteContextNode.inTensors = {&bmmVout};
  permuteContextNode.outTensors = {&context};
  permuteContextNode.inTensorViewFuncs.resize(
      permuteContextNode.inTensors.size());
  permuteContextNode.inTensorViewFuncs[0] =
      [=](const AsdOps::SVector<int64_t> &oldDims,
          AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.head_num,
                   param_.head_num,
                   oldDims.at(1),
                   oldDims.at(2)};
      };
}

SelfAttentionWithoutKvCacheOpsGPT3Runner::~SelfAttentionWithoutKvCacheOpsGPT3Runner() {}
}  // namespace AclTransformer
