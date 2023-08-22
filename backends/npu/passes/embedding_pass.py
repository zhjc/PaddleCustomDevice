import paddle

@paddle.incubate.passes.ir.RegisterPass
def generate_embedding():
    def pattern(w, ids):
        embedding_op = paddle.incubate.passes.ir.PassDesc.OP.lookup_table_v2
        return embedding_op(W=w, Ids=ids)

    def replace(w, ids):
        out = paddle.incubate.passes.ir.PassDesc.OP.embedding_op(W=w, Ids=ids)
        out.Attr("padding_idx").MappedPattern( op="lookup_table_v2", name="padding_idx")
        return out

    return pattern, replace
