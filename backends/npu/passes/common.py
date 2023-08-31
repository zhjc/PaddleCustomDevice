from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

from .split_pass import generate_split
from .ffn_pass import generate_ffn
from .add_norm_pass import generate_add_norm
from .layer_norm_pass import generate_layer_norm
from .matmul_pass import generate_matmul
from .attention_pass import gen_fuse_multi_head_attention, gen_gpt3_multi_head_attention
from .linear_pass import generate_linear
from .pad2d_pass import generate_pad2d
from .gpt3_layer_pass import (gen_fuse_attention_layer, gen_fuse_attention_cached_layer,
                              gen_fuse_attention_parallel_layer, gen_fuse_attention_cached_parallel_layer,
                              gen_fuse_attention_cached_parallel_async_layer)
from .embedding_pass import generate_embedding
from .identity_pass import generate_identity
from .embedding_add_pass import generate_embedding_add
from .dropout_pass import remove_dropout_after_ele_add
from .norm_matmul_pass import generate_norm_matmul

paddle.enable_static()

def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )

def addPasses(pass_builder):
    # pass_builder.append_pass("gen_gpt3_multi_head_attention")
    pass_builder.append_pass("gen_fuse_multi_head_attention")
    # pass_builder.append_pass("generate_split")
    pass_builder.append_pass("generate_ffn")
    pass_builder.append_pass("generate_add_norm")
    # pass_builder.append_pass("generate_matmul")
    pass_builder.append_pass("generate_linear")
    pass_builder.append_pass("generate_pad2d")
    #pass_builder.append_pass("generate_embedding")
    #pass_builder.append_pass("generate_embedding_add")
    paddle.fluid.core.register_subgraph_pass("gen_fuse_multi_head_attention")
    # paddle.fluid.core.register_subgraph_pass("generate_split")
    paddle.fluid.core.register_subgraph_pass("generate_ffn")
    paddle.fluid.core.register_subgraph_pass("generate_add_norm")
    # paddle.fluid.core.register_subgraph_pass("generate_matmul")
    paddle.fluid.core.register_subgraph_pass("generate_linear")
    paddle.fluid.core.register_subgraph_pass("generate_pad2d")
