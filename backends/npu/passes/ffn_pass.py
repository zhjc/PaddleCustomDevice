from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_ffn():
    def pattern(x, y, z):
        geluNode = paddle.incubate.passes.ir.PassDesc.OP.gelu(X=x)
        matmulNode = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2(X=geluNode, Y=y)
        return matmulNode

    def replace(x, y, z):
        FfnNode = paddle.incubate.passes.ir.PassDesc.OP.Ffn_pass_test(X=x, Y=y)
        return FfnNode

    return pattern, replace
