import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_identity():
    def pattern(x):
        identity_op = paddle.incubate.passes.ir.PassDesc.OP.c_identity
        identity_op.SetAttr("use_calc_stream", 1)
        identity_op.SetAttr("use_model_parallel", 1)
        identity_op.SetAttr("with_quant_attr", 0)
        return identity_op(X=x)

    def replace(x):
        return x

    return pattern, replace
