from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_layer_normal():
    def pattern(x, scale, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.layer_norm(X=z, Scale=scale, Bias=bias)

    def replace(x, weight, bias):
        layer_norm = paddle.incubate.passes.ir.PassDesc.OP.custom_layer_norm(X=z, Scale=scale, Bias=bias)
        layer_norm.SetAttr("begin_norm_axis", x.Attr("shape").Size() - 1)
        layer_norm.Attr("epsilon").MappedPattern(op="scale", name="bias")
        return layer_norm

    return pattern, replace
