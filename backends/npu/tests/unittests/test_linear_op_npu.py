import paddle
import numpy as np

paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op('/opt/py37env/lib/python3.7/site-packages/paddle_custom_device/libpaddle-custom-npu.so')

@paddle.incubate.passes.ir.RegisterPass
def generate_linear():
    def pattern(x, y, z):
        return paddle.add(paddle.matmul(x, y), z)

    def replace(x, y, z):
        return paddle.incubate.passes.ir.PassDesc.OP.linear(Input=x, Weight=y, Bias=z)

    return pattern, replace

@paddle.jit.to_static(input_spec=[paddle.static.InputSpec([2, 3, 6], 'float32', 'input'),  paddle.static.InputSpec([6, 18], 'float32', 'weight'),  paddle.static.InputSpec([18], 'float32', 'bias')])
def func(x, y, z):
    return paddle.add(paddle.matmul(x, y), z)

model_file = './saved_models/func'
paddle.jit.save(func, model_file)

# inference
config = paddle.inference.Config()
config.set_prog_file(model_file + '.pdmodel')
config.enable_memory_optim()
config.enable_custom_device("npu")
pass_builder = config.pass_builder()
pass_builder.append_pass('generate_linear')
print(pass_builder.all_passes())
predictor = paddle.inference.create_predictor(config)

input_names = predictor.get_input_names()
print(f"input_names={input_names}")
input_tensor = predictor.get_input_handle('input')
input_data = np.random.randn(2, 3, 6).astype('float32')
input_tensor.copy_from_cpu(input_data)
weight_tensor = predictor.get_input_handle('weight')
weight_data = np.random.randn(6, 18).astype('float32')
weight_tensor.copy_from_cpu(weight_data)
bias_tensor = predictor.get_input_handle('bias')
bias_data = np.random.randn(18).astype('float32')
bias_tensor.copy_from_cpu(bias_data)

predictor.run()

results = []
output_names = predictor.get_output_names()
print(f"output_names={output_names}")
for i, name in enumerate(output_names):
    output_tensor = predictor.get_output_handle(name)
    output_data = output_tensor.copy_to_cpu()
    results.append(output_data)

breakpoint()

print(results)
print(paddle.add(paddle.matmul(paddle.to_tensor(input_data), paddle.to_tensor(weight_data)), paddle.to_tensor(bias_data)))
