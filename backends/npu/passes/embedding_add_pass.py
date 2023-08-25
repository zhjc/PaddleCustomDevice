import paddle
@paddle.incubate.passes.ir.RegisterPass
def generate_embedding_add():
    def pattern(w0, ids0, w1, ids1):
        embedding_0 = paddle.incubate.passes.ir.PassDesc.OP.lookup_table_v2(W=w0, Ids=ids0)
        embedding_1 = paddle.incubate.passes.ir.PassDesc.OP.lookup_table_v2(W=w1, Ids=ids1)
        add = paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=embedding_0, Y=embedding_1)
        dropout_op = paddle.incubate.passes.ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        return dropout_op(X=add)

    def replace(w0, ids0, w1, ids1):
        out = paddle.incubate.passes.ir.PassDesc.OP.pretreatment_op(W0=w0, Ids0=ids0, W1=w1, Ids1=ids1)
        out.Attr("padding_idx0").MappedPattern(op="lookup_table_v2", name="padding_idx")
        out.Attr("padding_idx1").MappedPattern(op="lookup_table_v2", name="padding_idx")
        return out

    return pattern, replace
