from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_ffn():
    def pattern(x, y):
        return paddle.nn.functional.gelu(paddle.add(paddle.matmul(x, y), z))

    def replace(x, y):
        FfnNode = paddle.incubate.passes.ir.PassDesc.OP.ffn(X=x, Y=y)
        return FfnNode

    return pattern, replace
