from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_matmul():
    def pattern(x, y):
        matmul_out = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2(X=x, Y=y)
        return matmul_out

    def replace(x, y):
        return paddle.incubate.passes.ir.PassDesc.OP.add_norm(X=x, Y=x, Weight=y, Bias=y)

    return pattern, replace
