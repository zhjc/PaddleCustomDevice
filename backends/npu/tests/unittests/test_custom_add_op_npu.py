import paddle
import numpy as np

paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op('/opt/py37env/lib/python3.7/site-packages/paddle_custom_device/libpaddle-custom-npu.so')

@paddle.incubate.passes.ir.RegisterPass
def generate_custom_add():
    def pattern(x, y):
        return paddle.add(x, y)

    def replace(x, y):
        return paddle.incubate.passes.ir.PassDesc.OP.custom_add(X=x, Y=y)

    return pattern, replace

@paddle.jit.to_static(input_spec=[paddle.static.InputSpec([3, 9], 'float32', 'x'),  paddle.static.InputSpec([3, 9], 'float32', 'y')])
def func(x, y):
    return paddle.add(x, y)

model_file = './saved_models/func'
paddle.jit.save(func, model_file)

# inference
config = paddle.inference.Config()
config.set_prog_file(model_file + '.pdmodel')
config.enable_memory_optim()
config.enable_custom_device("npu")
pass_builder = config.pass_builder()
pass_builder.append_pass('generate_custom_add')
print(pass_builder.all_passes())
predictor = paddle.inference.create_predictor(config)

input_names = predictor.get_input_names()
print(f"input_names={input_names}")
x_tensor = predictor.get_input_handle('x')
x_data = np.random.randn(3, 9).astype('float32')
x_tensor.copy_from_cpu(x_data)
y_tensor = predictor.get_input_handle('y')
y_data = np.random.randn(3, 9).astype('float32')
y_tensor.copy_from_cpu(y_data)

predictor.run()

results = []
output_names = predictor.get_output_names()
print(f"output_names={output_names}")
for i, name in enumerate(output_names):
    output_tensor = predictor.get_output_handle(name)
    output_data = output_tensor.copy_to_cpu()
    results.append(output_data)

print(results)
print([paddle.add(paddle.to_tensor(x_data), paddle.to_tensor(y_data)).numpy()])
