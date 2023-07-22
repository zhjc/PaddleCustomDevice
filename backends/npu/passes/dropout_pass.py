from paddle.incubate.passes import ir

@paddle.incubate.passes.ir.RegisterPass
def remove_dropout_after_ele_add():
    def pattern(x, y):
        x_plus_y = ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        return dropout_op(X=x_plus_y)
    
    def replace(x, y):
        return ir.PassDesc.OP.elementwise_add(X=x, Y=y)
    
    return pattern, replace

@paddle.incubate.passes.ir.RegisterPass
def remove_dropout_after_softmax():
    def pattern(x):
        x = ir.PassDesc.OP.softmax(X=x)
        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        return dropout_op(X=x)
    
    def replace(x, y):
        return ir.PassDesc.OP.softmax(X=x)
    
    return pattern, replace