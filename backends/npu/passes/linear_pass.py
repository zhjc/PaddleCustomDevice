from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_linear():
    def pattern(x, weight, bias):
        matmul = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2(X=x, Y=weight)
        return paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=matmul, Y=bias)

    def replace(x, weight, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.linear(Input=x, Weight=w, Bias=bias)

    return pattern, replace
