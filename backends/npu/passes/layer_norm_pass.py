from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_layer_norm():
    def pattern(x, scale, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.layer_norm(X=x,
            Scale=scale, Bias=bias).Output("Y")

    def replace(x, weight, bias):
        layer_norm = paddle.incubate.passes.ir.PassDesc.OP.custom_layer_norm(X=x, Scale=weight, Bias=bias)
        layer_norm.Attr("begin_norm_axis").MappedPattern(op="layer_norm", name="begin_norm_axis")
        layer_norm.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon")
        return layer_norm

    return pattern, replace
