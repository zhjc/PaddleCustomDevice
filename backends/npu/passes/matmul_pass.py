from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_matmul():
    def pattern(x, y):
        matmul_out = paddle.incubate.passes.ir.PassDesc.OP.matmul_v2(X=x, Y=y)
        return matmul_out

    def replace(x, y):
        matmul = paddle.incubate.passes.ir.PassDesc.OP.custom_matmul(X=x, Y=y)
        matmul.Attr("trans_x").MappedPattern(op="matmul_v2", name="trans_x")
        matmul.Attr("trans_y").MappedPattern(op="matmul_v2", name="trans_y")
        return matmul
    return pattern, replace
