from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_split():
    def pattern(x, y, z):
        splitNode = paddle.incubate.passes.ir.PassDesc.OP.split(X=x)
        splitNode.SetAttr("num", 3)
        return matmulNode

    def replace(x, y, z):
        newSplitNode = paddle.incubate.passes.ir.PassDesc.OP.split_pass_test(X=x, Y=y)
        newSplitNode.SetAttr("num", 3)
        return newSplitNode

    return pattern, replace
