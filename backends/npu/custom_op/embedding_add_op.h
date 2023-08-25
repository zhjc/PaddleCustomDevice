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

#ifndef EMBEDDING_ADD_OP_H
#define EMBEDDING_ADD_OP_H

#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "acltransformer/graph_operation.h"

struct PretreatmentWorkspace {
  void *workspace_ = nullptr;
  uint64_t workspaceSize_ = 0;
};

namespace AclTransformer {
struct PretreatmentParam {
  int axis0 = 0;
  int axis1 = 0;
};

class PretreatmentOperation : public GraphOperation {
 public:
  explicit PretreatmentOperation(const PretreatmentParam &param);
  ~PretreatmentOperation();
  uint64_t GetInTensorCount() const override;
  uint64_t GetOutTensorCount() const override;

 protected:
  AsdOps::Status InferShapeImpl(
      const AsdOps::SVector<AsdOps::Tensor> &inTensors,
      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

 private:
  PretreatmentParam param_;
};
}  // namespace AclTransformer
#endif // PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#endif