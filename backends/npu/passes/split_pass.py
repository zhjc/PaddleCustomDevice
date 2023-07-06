from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_split():
    def pattern(x):
        splitNode = paddle.incubate.passes.ir.PassDesc.OP.split(X=x)
        splitNode.SetAttr("num", 3)
        return splitNode

    def replace(x):
        newSplitNode = paddle.incubate.passes.ir.PassDesc.OP.split_pass_test(X=x)
        newSplitNode.SetAttr("num", 3)
        return newSplitNode

    return pattern, replace
