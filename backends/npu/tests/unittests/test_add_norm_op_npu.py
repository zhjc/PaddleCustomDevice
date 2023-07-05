import paddle
import numpy as np

paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op('/opt/py37env/lib/python3.7/site-packages/paddle_custom_device/libpaddle-custom-npu.so')

@paddle.incubate.passes.ir.RegisterPass
def generate_add_norm():
    def pattern(x, y, weight, bias):
        z = paddle.incubate.passes.ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        layer_norm_res = paddle.incubate.passes.ir.PassDesc.OP.layer_norm(X=z, Scale=weight, Bias=bias)
        return layer_norm_res.Output("Y")

    def replace(x, y, weight, bias):
        return paddle.incubate.passes.ir.PassDesc.OP.add_norm(X=x, Y=y, Weight=weight, Bias=bias)

    return pattern, replace

@paddle.jit.to_static(input_spec=[paddle.static.InputSpec([1, 32, 128], 'float32', 'x'), paddle.static.InputSpec([1, 32, 128], 'float32', 'y'), paddle.static.InputSpec([128,], 'float32', 'weight'),  paddle.static.InputSpec([128,], 'float32', 'bias')])
def func(x, y, weight, bias):
    return paddle.nn.functional.layer_norm(paddle.add(x, y), x.shape[-1], weight=weight, bias=bias)

model_file = './saved_models/func'
paddle.jit.save(func, model_file)

# inference
config = paddle.inference.Config()
config.set_prog_file(model_file + '.pdmodel')
config.enable_memory_optim()
config.enable_custom_device("npu")
pass_builder = config.pass_builder()
pass_builder.append_pass('generate_add_norm')
print(f"[INFO]pass_builder.all_passes()={pass_builder.all_passes()}")
predictor = paddle.inference.create_predictor(config)

input_names = predictor.get_input_names()
print(f"[INFO]input_names={input_names}")
x_tensor = predictor.get_input_handle('x')
x_data = np.random.randn(1, 32, 128).astype('float32')
x_tensor.copy_from_cpu(x_data)
y_tensor = predictor.get_input_handle('y')
y_data = np.random.randn(1, 32, 128).astype('float32')
y_tensor.copy_from_cpu(y_data)
weight_tensor = predictor.get_input_handle('weight')
weight_data = np.random.randn(128).astype('float32')
weight_tensor.copy_from_cpu(weight_data)
bias_tensor = predictor.get_input_handle('bias')
bias_data = np.random.randn(128).astype('float32')
bias_tensor.copy_from_cpu(bias_data)

predictor.run()

results = []
output_names = predictor.get_output_names()
print(f"[INFO]output_names={output_names}")
for i, name in enumerate(output_names):
    output_tensor = predictor.get_output_handle(name)
    output_data = output_tensor.copy_to_cpu()
    results.append(output_data)

print(results)
print(paddle.nn.functional.layer_norm(paddle.add(paddle.to_tensor(x_data), paddle.to_tensor(y_data)), x_data.shape[-1], weight=paddle.to_tensor(weight_data), bias=paddle.to_tensor(bias_data)))
print(paddle.add(paddle.to_tensor(x_data), paddle.to_tensor(y_data)))

