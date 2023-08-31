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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <iostream>
#include <vector>
#include "runtime/runtime.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

#include "gpt3_layer_op.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/plan.h"
#include "acltransformer/statistic.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/cast_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "self_attention_kv_cache_fusion_gpt3/self_attention_kv_cache_fusion_gpt3_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "kernels/funcs/format_utils.h"

namespace AclTransformer
{
enum GPT3LayerDecoderTensorId
{
  IN_HIDDENSTATES = 0,
  IN_NORMWEIGHT,
  IN_NORMBIAS,
  IN_QKVMIXDWEIGHT,
  IN_QKVMIXDBIAS,
  IN_SELFOUTLINEARWEIGHT,
  IN_SELFOUTLINEARBIAS,
  IN_SELFOUTNORMWEIGHT,
  IN_SELFOUTNORMBIAS,
  IN_FFNLINEARWEIGHT,
  IN_FFNLINEARBIAS,
  IN_FFNOUTLINEARWEIGHT,
  IN_FFNOUTLINEARBIAS,
  IN_SEQLEN,
  IN_TOKENOFFSET,
  IN_LAYERID,
  IN_ATTENTIONMASK,
  IN_PASTKEY,
  IN_PASTVALUE,

  OUT_GPT3LAYEROUT,

  INTERMIDATE_INPUTNORMOUT,
  INTERMIDATE_MIXEDLINEAROUTQKV,

  INTERMIDATE_SELFOUT,
  INTERMIDATE_SELFLINEAROUT,
  INTERMIDATE_SELFRESIDUALADDOUT,
  
  INTERMIDATE_SELFNORMOUT,
  INTERMIDATE_FFNOUT,
  INTERMIDATE_FFNLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

GPT3LayerDecoderOperation::GPT3LayerDecoderOperation(const GPT3LayerParam &param)
    : GraphOperation("GPT3LayerDecoderOperation"), param_(param)
{
  opGraph_.inTensorSize = IN_TENSOR_COUNT;
  opGraph_.outTensorSize = OUT_TENSOR_COUNT;
  opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
  opGraph_.nodes.resize(NODE_COUNT);

  size_t nodeId = 0;
  GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnLinearNode = opGraph_.nodes.at(nodeId++);
  GraphOperation::Node &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

  inputNormNode.operation.reset(new AclTransformer::NormOperation(
      {param_.layerNormEps, param_.layerNormBeginNormAxis, param_.layerNormBeginNormAxis}));
  inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
  inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

  mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true})); /* 加速库默认会将w进行转置 */
  mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
  mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

  selfAttentionKvCacheNode.operation.reset(new AclTransformer::SelfAttentionKvCacheFusionGPT3Operation(
      {param_.head_num, param_.layer_num - 1, param_.head_dim, 0, 0, 0, "chatglm6b", param_.seqLen, param_.tokenOffset}));
  selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV,
                                          IN_PASTKEY,
                                          IN_PASTVALUE,
                                          IN_ATTENTIONMASK,
                                          IN_TOKENOFFSET,
                                          IN_SEQLEN,
                                          IN_LAYERID};
  selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
  selfAttentionKvCacheNode.useVariantPackParam = true;

  selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true}));
  selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
  selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

  selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
  selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

  selfNormNode.operation.reset(new AclTransformer::NormOperation(
      {param_.layerNormEps, param_.layerNormBeginNormAxis, param_.layerNormBeginNormAxis}));
  selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
  selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

  ffnNode.operation.reset(new AclTransformer::FfnOperation({false, true}));
  ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
  ffnNode.outTensorIds = {INTERMIDATE_FFNOUT};

  ffnLinearNode.operation.reset(new AclTransformer::LinearOperation({false, true}));
  ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
  ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT};

  ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
  ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_FFNLINEAROUT};
  ffnResidualAddNode.outTensorIds = {OUT_GPT3LAYEROUT};
}

GPT3LayerDecoderOperation::~GPT3LayerDecoderOperation() {}

uint64_t GPT3LayerDecoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GPT3LayerDecoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GPT3LayerDecoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{

  outTensorDescs.at(0) = inTensors.at(0).desc;

  return AsdOps::Status::OkStatus();
}

} // namespace AclTransformer

GPT3LayerWorkspace g_gpt3WorkSpace = {nullptr, 0};

std::unique_ptr<AclTransformer::GPT3LayerDecoderOperation> g_gpt3DecoderOp;
std::unique_ptr<AclTransformer::Plan> g_gpt3Plan;

AclTransformer::SelfAttentionKvCacheFusionVariantPackParam g_variantPackParam_;
AsdOps::Tensor g_gpt3_cachek;
AsdOps::Tensor g_gpt3_cachev;
AsdOps::Tensor g_gpt3_attenmask;

// std::vector<int32_t> g_seq_len_vector(1, 1); /* 增量的q_seq_len，为1，当前也只考虑batch为1 */
// AsdOps::SVector<int32_t> g_seq_len(1, 1);
std::vector<int32_t> g_token_offset_vector;
AsdOps::SVector<int32_t> g_token_offset = {1};

std::vector<std::vector<int64_t>> GPT3LayerOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &norm_bias_shape,
    const std::vector<int64_t> &mix_linear_weight_shape,
    const std::vector<int64_t> &mix_linear_bias_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_linear_bias_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &self_out_norm_bias_shape,
    const std::vector<int64_t> &ffn_linear_weight_shape,
    const std::vector<int64_t> &ffn_linear_bias_shape,
    const std::vector<int64_t> &ffn_out_linear_weight_shape,
    const std::vector<int64_t> &ffn_out_linear_bias_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &pastkey_shape,
    const std::vector<int64_t> &pastvalue_shape)
{
  return {hidden_shape, pastkey_shape, pastvalue_shape};
}

static void BuildVariantPack(AsdOps::SVector<const phi::DenseTensor *> &inTensors,
                             phi::DenseTensor &outTensors,
                             AclTransformer::VariantPack &variantPack)
{
  variantPack.inTensors.resize(inTensors.size() + 3);
  for (size_t i = 0; i < inTensors.size(); ++i)
  {
    variantPack.inTensors.at(i) = ConvertDenseTensorToAsdTensor(*(inTensors.at(i)));
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
        variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
      variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
    }
  }
  variantPack.inTensors.at(inTensors.size()) = g_gpt3_attenmask;
  variantPack.inTensors.at(inTensors.size() + 1) = g_gpt3_cachek;
  variantPack.inTensors.at(inTensors.size() + 2) = g_gpt3_cachev;

  variantPack.outTensors.resize(1);
  variantPack.outTensors.at(0) = ConvertDenseTensorToAsdTensor(outTensors);
  if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
      variantPack.outTensors.at(0).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
    variantPack.outTensors.at(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
  }
}

static void SetWorkspace(uint64_t workspaceSize)
{
  if (workspaceSize <= g_gpt3WorkSpace.workspaceSize_) {
    VLOG(4) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
            << " <= workspaceSize_:" << g_gpt3WorkSpace.workspaceSize_ << ", not new device mem";
    return;
  }

  if (g_gpt3WorkSpace.workspace_) {
    AsdRtMemFreeDevice(g_gpt3WorkSpace.workspace_);
    g_gpt3WorkSpace.workspace_ = nullptr;
    g_gpt3WorkSpace.workspaceSize_ = 0;
  }

  int st = AsdRtMemMallocDevice((void **)&(g_gpt3WorkSpace.workspace_), workspaceSize, ASDRT_MEM_DEFAULT);

  PADDLE_ENFORCE_EQ(st,
                    ASDRT_SUCCESS,
                    phi::errors::External(
                        "GPT3LayerOp SetWorkspace AsdRtMemMallocDevice,"
                        "fail, ret: %d .",
                        st));

  g_gpt3WorkSpace.workspaceSize_ = workspaceSize;
}

static void *GetWorkspace() { return g_gpt3WorkSpace.workspace_; }

void GPT3LayerGetTensorInputs(
    const phi::CustomContext &dev_ctx,
    uint32_t layer_id,
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_bias,
    const paddle::Tensor &mix_linear_weight,
    const paddle::Tensor &mix_linear_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_bias,
    const paddle::Tensor &ffn_linear_weight,
    const paddle::Tensor &ffn_linear_bias,
    const paddle::Tensor &ffn_out_linear_weight,
    const paddle::Tensor &ffn_out_linear_bias,
    phi::DenseTensor &seq_len_dense,
    phi::DenseTensor &token_offset_dense,
    phi::DenseTensor &layer_id_dense,
    AsdOps::SVector<const phi::DenseTensor *> &inputs)
{
  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_bias_tensor = static_cast<const phi::DenseTensor *>(norm_bias.impl().get());
  auto mix_linear_weight_tensor = static_cast<const phi::DenseTensor *>(mix_linear_weight.impl().get());
  auto mix_linear_bias_tensor = static_cast<const phi::DenseTensor *>(mix_linear_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_linear_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_bias.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_bias.impl().get());
  auto ffn_linear_weight_tensor = static_cast<const phi::DenseTensor *>(ffn_linear_weight.impl().get());
  auto ffn_linear_bias_tensor = static_cast<const phi::DenseTensor *>(ffn_linear_bias.impl().get());
  auto ffn_out_linear_weight_tensor = static_cast<const phi::DenseTensor *>(ffn_out_linear_weight.impl().get());
  auto ffn_out_linear_bias_tensor = static_cast<const phi::DenseTensor *>(ffn_out_linear_bias.impl().get());

  std::vector<int32_t> layer_id_vec(1, layer_id);
  std::vector<int32_t> g_seq_len_vector(hidden.shape().at(0), 1);
  custom_kernel::TensorFromVector(dev_ctx, g_seq_len_vector, dev_ctx, &seq_len_dense);
  custom_kernel::TensorFromVector(dev_ctx, g_token_offset_vector, dev_ctx, &token_offset_dense);
  custom_kernel::TensorFromVector(dev_ctx, layer_id_vec, dev_ctx, &layer_id_dense);

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_bias_tensor);
  inputs.push_back(mix_linear_weight_tensor);
  inputs.push_back(mix_linear_bias_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_bias_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_bias_tensor);
  inputs.push_back(ffn_linear_weight_tensor);
  inputs.push_back(ffn_linear_bias_tensor);
  inputs.push_back(ffn_out_linear_weight_tensor);
  inputs.push_back(ffn_out_linear_bias_tensor);
  inputs.push_back(&seq_len_dense);
  inputs.push_back(&token_offset_dense);
  inputs.push_back(&layer_id_dense);
}

void InitFlashAttentionTensor(int layer_num,
                              int batch,
                              int org_seq_len,
                              int max_seqlen,
                              int head_dim,
                              int head_num,
                              const phi::DenseTensor *cache_k,
                              const phi::DenseTensor *cache_v)
{
  /* [layer_num, batch, max_seqlen, hidden_size] */
  uint64_t cache_k_size = batch * max_seqlen * head_dim * head_num * layer_num * sizeof(std::uint16_t); /* 应该是float16 */

  g_gpt3_cachek.desc = {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {layer_num, batch, max_seqlen, head_dim * head_num}};
  g_gpt3_cachek.dataSize = cache_k_size;
  int st = AsdRtMemMallocDevice((void **)&(g_gpt3_cachek.data), cache_k_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(st,
                    ASDRT_SUCCESS,
                    phi::errors::External(
                        "Alloc g_gpt3_cachek AsdRtMemMallocDevice,"
                        "fail, ret: %d size: %d.",
                        st, cache_k_size));
  AsdRtMemCopy(g_gpt3_cachek.data, cache_k_size, const_cast<void *>(cache_k->data()), cache_k_size, ASDRT_MEMCOPY_HOST_TO_DEVICE);

  g_gpt3_cachev.desc = {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {layer_num, batch, max_seqlen, head_dim * head_num}};
  g_gpt3_cachev.dataSize = cache_k_size;
  st = AsdRtMemMallocDevice((void **)&(g_gpt3_cachev.data), cache_k_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(st,
                    ASDRT_SUCCESS,
                    phi::errors::External(
                        "Alloc g_gpt3_cachev AsdRtMemMallocDevice,"
                        "fail, ret: %d size: %d.",
                        st, cache_k_size));
  AsdRtMemCopy(g_gpt3_cachev.data, cache_k_size, const_cast<void *>(cache_v->data()), cache_k_size, ASDRT_MEMCOPY_HOST_TO_DEVICE);

  uint64_t atten_size = max_seqlen * max_seqlen * sizeof(std::uint16_t);
  g_gpt3_attenmask.desc = {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {max_seqlen, max_seqlen}};
  g_gpt3_attenmask.dataSize = atten_size;
  st = AsdRtMemMallocDevice((void **)&(g_gpt3_attenmask.data), atten_size, ASDRT_MEM_DEFAULT);
  PADDLE_ENFORCE_EQ(st,
                    ASDRT_SUCCESS,
                    phi::errors::External(
                        "Alloc g_gpt3_attenmask AsdRtMemMallocDevice,"
                        "fail, ret: %d size: %d.",
                        st, atten_size));
}

std::vector<paddle::Tensor> GPT3LayerOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_bias,
    const paddle::Tensor &mix_linear_weight,
    const paddle::Tensor &mix_linear_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_bias,
    const paddle::Tensor &ffn_linear_weight,
    const paddle::Tensor &ffn_linear_bias,
    const paddle::Tensor &ffn_out_linear_weight,
    const paddle::Tensor &ffn_out_linear_bias,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &past_key,
    const paddle::Tensor &past_value,
    int begin_norm_axis,
    float epsilon,
    std::vector<int32_t> shape,
    float scale) {

  AsdOps::Timer timer;
  
  int32_t layer_num = (int32_t)scale;
  int32_t head_dim = shape[3] / 3;
  int32_t head_num = hidden.shape()[2] / head_dim;

  static uint32_t layerId = 0;

  if (layerId % layer_num == 0) { /* 0 ....31,第2个token32次时候清零，调整 */
    layerId = 0;
    if (!g_gpt3DecoderOp)
    { /* token_offset为kvLen，第一个token，初始化为org_seq_len */

      int org_seq_len = past_key.shape().at(1);
      int batch_tmp = past_key.shape().at(0);
      g_token_offset_vector.clear();
      g_token_offset_vector.resize(batch_tmp, org_seq_len);
      AsdOps::SVector<int32_t> g_token_offset_t(batch_tmp, org_seq_len);
	    g_token_offset = g_token_offset_t;
      // g_token_offset.clear();
      // g_token_offset.resize(batch_tmp);
      // g_token_offset.push_back(org_seq_len);
    } else { /* 每处理个token，长度加1 */
      std::for_each(g_token_offset_vector.begin(), g_token_offset_vector.end(), [](int &n)
                    { n++; });
      std::for_each(g_token_offset.begin(), g_token_offset.end(), [](int &n)
                    { n++; });
    }
  }

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  AclTransformer::Handle handle = {stream};

  phi::DenseTensor seq_len_tensor;
  phi::DenseTensor token_offset_tensor;
  phi::DenseTensor layer_id_tensor;
  AsdOps::SVector<const phi::DenseTensor *> inputs;

  GPT3LayerGetTensorInputs(*dev_ctx,
                           layerId,
                           hidden,
                           norm_weight,
                           norm_bias,
                           mix_linear_weight,
                           mix_linear_bias,
                           self_out_linear_weight,
                           self_out_linear_bias,
                           self_out_norm_weight,
                           self_out_norm_bias,
                           ffn_linear_weight,
                           ffn_linear_bias,
                           ffn_out_linear_weight,
                           ffn_out_linear_bias,
                           seq_len_tensor,
                           token_offset_tensor,
                           layer_id_tensor,
                           inputs);
  layerId++;

  std::shared_ptr<phi::DenseTensor> gpt3layerout_tensor =
      std::make_shared<phi::DenseTensor>();
  gpt3layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(gpt3layerout_tensor.get(), inputs.at(0)->dtype());
  
  AsdOps::SVector<int32_t> g_seq_len(past_key.shape().at(0), 1);
  
  if (!g_gpt3DecoderOp) {
    int batch_tmp = past_key.shape().at(0);
    int max_seq_len_tmp = 1024;
    int org_seq_len = past_key.shape().at(1);
    InitFlashAttentionTensor(layer_num, batch_tmp, org_seq_len, max_seq_len_tmp,
        head_dim, head_num, inputs[inputs.size() - 2], inputs[inputs.size() - 1]);
    AclTransformer::GPT3LayerParam param =
        {epsilon, begin_norm_axis, head_dim, head_num, layer_num, g_token_offset, g_seq_len};
    g_gpt3DecoderOp.reset(new AclTransformer::GPT3LayerDecoderOperation(param));
    g_gpt3Plan.reset(new AclTransformer::Plan);
    g_gpt3DecoderOp->BuildPlan(g_gpt3Plan.get());
  }

  static uint64_t executeCount_ = 0;

  AclTransformer::VariantPack variantPack;
  g_variantPackParam_ = {g_seq_len, g_token_offset};
  variantPack.param = g_variantPackParam_;
  BuildVariantPack(inputs, *gpt3layerout_tensor.get(), variantPack);

  /* Set up */
  AsdOps::Timer timer1;
  AsdOps::Status st = g_gpt3Plan->Setup(handle, variantPack);

  AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External(
                        "GPT3LayerOp Setup plan failed,"
                        "ret message: %s .",
                        st.Message()));

  variantPack.workspaceSize = g_gpt3Plan->GetWorkspaceSize();
  VLOG(4) << " GPT3LayerOp plan workspace size:" << variantPack.workspaceSize;

  if (variantPack.workspaceSize > 0) {
    SetWorkspace(variantPack.workspaceSize);
    variantPack.workspace = GetWorkspace();
  }

  /* Execute */
  AsdOps::Timer timer2;
  st = g_gpt3Plan->Execute(handle, variantPack);

  AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
  PADDLE_ENFORCE_EQ(st.Ok(),
                    true,
                    phi::errors::External(
                        "GPT3LayerOp Execute plan failed,"
                        "ret message: %s .",
                        st.Message()));
  AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer.ElapsedMicroSecond();
  ASD_LOG(FATAL) << GPT3LayerOp << " executeCount:" << executeCount_++ << ", statistic:["
                 << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
  AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

  if ((executeCount_) % layer_num == 0) { // 1.....32,第32次同步
    int ret = aclrtSynchronizeStream(stream);
  }  
  
  return {paddle::Tensor(gpt3layerout_tensor), past_key, past_value};
}

PD_BUILD_OP(gpt3_layer)
    .Inputs({"Hidden",
             "NormWeight",
             "NormBias",
             "MixLinearWeight",
             "MixLinearBias",
             "SelfOutLinearWeight",
             "SelfOutLinearBias",
             "SelfOutNormWeight",
             "SelfOutNormBias",
             "FfnLinearWeight",
             "FfnLinearBias",
             "FfnOutLinearWeight",
             "FfnOutLinearBias",
             "AttentionMask",
             "PastKey",
             "PastValue"})
    .Outputs({"Out", "PresentKey", "PresentValue"})
    .Attrs({"begin_norm_axis: int",
            "epsilon: float",
            "shape: std::vector<int>",
            "scale: float"})
    .SetKernelFn(PD_KERNEL(GPT3LayerOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        GPT3LayerOpInferShape)); // neccessary if the op has muti_inputs
#endif
