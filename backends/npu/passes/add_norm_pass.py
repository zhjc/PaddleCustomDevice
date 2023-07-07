from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_add_norm():
    def pattern(x, y, weight, bias):
        z = paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        layer_norm_out = paddle.incubate.passes.ir.PassDesc.OP.layer_norm(X=z, Scale=weight, Bias=bias)
        return layer_norm_out.Output("Y")

    def replace(x, y, weight, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.add_norm(X=x, Y=y, Weight=weight, Bias=bias)

    return pattern, replace
