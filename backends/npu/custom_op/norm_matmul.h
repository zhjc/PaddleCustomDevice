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

#ifndef ACLTRANSTORMOR_NORM_MATMUL_OP_H
#define ACLTRANSTORMOR_NORM_MATMUL_OP_H

#include "acltransformer/graph_operation.h"

struct NormMatmulWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

namespace AclTransformer {
struct NormMatmulParam {
  float layerNormEps = 1e-5;
  int layerNormBeginNormAxis = 2;
  bool trans_x = false;
  bool trans_y = true;
};

class NormMatmulOperation : public GraphOperation {
 public:
  explicit NormMatmulOperation(const NormMatmulParam &param);
  ~NormMatmulOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(
      const AsdOps::SVector<AsdOps::Tensor> &inTensors,
      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  NormMatmulParam param_;
};
}  // namespace AclTransformer
#endif