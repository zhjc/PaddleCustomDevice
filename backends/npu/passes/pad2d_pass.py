from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_pad2d():
    def pattern(x):
        pad3d_in = paddle.incubate.passes.ir.PassDesc.OP.unsqueeze2(X=x)
        pad3d_out = paddle.incubate.passes.ir.PassDesc.OP.pad3d(X=pad3d_in.Output("Out"))
        res = paddle.incubate.passes.ir.PassDesc.OP.squeeze2(X=pad3d_out)
        return res.Output("Out")

    def replace(x):
        return paddle.incubate.passes.ir.PassDesc.OP.pad2d(X=x)

    return pattern, replace