import paddle
def layer_norm_only_out(x, scale, bias):
    layernorm_op = paddle.incubate.passes.ir.PassDesc.OP.layer_norm
    layernorm_op._outputs.pop("Mean")
    layernorm_op._outputs.pop("Variance")
    x = layernorm_op(X=x, Scale=scale, Bias=bias)
    return x


@paddle.incubate.passes.ir.RegisterPass
def generate_norm_matmul():
    def pattern(x, scale, bias, y):
        norm = layer_norm_only_out(x, scale, bias)
        matmul_op = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2
        matmul_op.SetAttr("trans_y", True)
        return matmul_op(X=norm, Y=y)

    def replace(x, scale, bias, y):
        out = paddle.incubate.passes.ir.PassDesc.OP.norm_matmul_op(X=x, Scale=scale, Bias=bias, Y=y)
        out.Attr("epsilon").MappedPattern(
            op="layer_norm", name="epsilon")
        out.Attr("begin_norm_axis").MappedPattern(
            op="layer_norm", name="begin_norm_axis")
        out.Attr("trans_x").MappedPattern(
            op="matmul_v2", name="trans_x")
        out.Attr("trans_y").MappedPattern( op="matmul_v2", name="trans_y")

        return out

    return pattern, replace
