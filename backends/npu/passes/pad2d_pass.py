from __future__ import print_function, division
import paddle


@paddle.incubate.passes.ir.RegisterPass
def generate_pad2d():
    def pattern(x):
        unsqueeze_op = paddle.incubate.passes.ir.PassDesc.OP.unsqueeze2(X=x)
        unsqueeze_op._outputs.pop("XShape")
        pad3d_in = unsqueeze_op(X=x)

        pad3d_out = paddle.incubate.passes.ir.PassDesc.OP.pad3d(X=pad3d_in)

        squeeze_op = paddle.incubate.passes.ir.PassDesc.OP.squeeze2(X=pad3d_out)
        squeeze_op._outputs.pop("XShape")
        res = squeeze_op(X=pad3d_out)

        return res.Output("Out")

    def replace(x):
        pad = paddle.incubate.passes.ir.PassDesc.OP.pad3d(X=x)
        pad.Attr("paddings").MappedPattern(op="pad3d", name="paddings")
        return pad

    return pattern, replace