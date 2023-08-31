from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

from paddle.incubate.passes import ir
import time

@paddle.incubate.passes.ir.RegisterPass
def gen_fuse_multi_head_attention():
    def transpose_without_shape(x):
        op = ir.PassDesc.OP.transpose2
        op.SetAttr("axis", [0, 2, 1, 3])
        op._outputs.pop("XShape")
        return op(X=x)
    
    def reshape_without_shape(x):
        op = ir.PassDesc.OP.reshape2
        op.SetAttr("shape", [0, 0, -1])
        op._outputs.pop("XShape")
        return op(X=x)
    
    def pattern(q, concated_k, concated_v, attn_mask):
        transed_q = transpose_without_shape(q)
        transed_k = transpose_without_shape(concated_k)
        transed_v = transpose_without_shape(concated_v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)

        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)

        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)

        out = transpose_without_shape(out)

        out = reshape_without_shape(out)

        return out

    def replace(q, concated_k, concated_v, attn_mask):
        reshape_q = reshape_without_shape(q)
        reshape_k = reshape_without_shape(concated_k)
        reshape_v = reshape_without_shape(concated_v)
        
        mask_tmp = ir.PassDesc.OP.unsqueeze2(X=attn_mask)
        mask_tmp.SetAttr("axes", [2])
        new_mask = ir.PassDesc.OP.gather_nd(X=mask_tmp.Output("Out"), Index=paddle.to_tensor([0]))
        
        return ir.PassDesc.OP.flash_attention(Query=reshape_q, CacheKey=reshape_k, CacheValue=reshape_v, AttentionMask=new_mask)


    return pattern, replace


@paddle.incubate.passes.ir.RegisterPass
def gen_gpt3_multi_head_attention():
    def transpose_without_shape(x):
        op = ir.PassDesc.OP.transpose2
        op.SetAttr("axis", [0, 2, 1, 3])
        op._outputs.pop("XShape")
        return op(X=x)
    
    def reshape_without_shape(x):
        op = ir.PassDesc.OP.reshape2
        op.SetAttr("shape", [0, 0, -1])
        op._outputs.pop("XShape")
        return op(X=x)
    
    def pattern(q, concated_k, concated_v, attn_mask):
        transed_q = transpose_without_shape(q)
        transed_k = transpose_without_shape(concated_k)
        transed_v = transpose_without_shape(concated_v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)

        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)

        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)

        out = transpose_without_shape(out)

        out = reshape_without_shape(out)

        return out
    
    def replace(q, concated_k, concated_v, attn_mask):
        attention_node = ir.PassDesc.OP.multihead_attention(Q=q, ConcatedK=concated_k, ConcatedV=concated_v, AttnMask=attn_mask)
        attention_node.SetAttr("layer_num")
        attention_node.Attr("layer_num").MappedPattern(op="scale", name="scale", layer_num = 1)
        return 

    return pattern, replace
