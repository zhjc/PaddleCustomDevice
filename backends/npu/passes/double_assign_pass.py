from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_assign():
    def pattern(x, y):
        return paddle.assign(x), paddle.assign(y)

    def replace(x, y, bias):
        return x, y

    return pattern, replace
