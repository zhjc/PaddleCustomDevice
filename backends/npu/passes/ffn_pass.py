from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_ffn():
    def pattern(x, y, bias):
        return paddle.nn.functional.gelu(paddle.add(paddle.matmul(x, y), bias))

    def replace(x, y, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.ffn(Input=x, Weight=y, Bias=bias)

    return pattern, replace
