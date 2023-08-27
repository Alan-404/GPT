#%%
import tensorrt as trt
import numpy as np
from preprocess import Tokenizer
import pycuda.driver as cuda
import pycuda.autoinit
import time

engine_path = './engines/gpt.trt'
tokenizer_path = './tokenizer/tokenizer.pkl'
message = "máy tính là gì?"
PRECISION = np.float32
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
max_ctx = 250
# %%
tokenizer = Tokenizer(tokenizer_path)
token_size = len(tokenizer.dictionary)
digits = tokenizer.text_to_sequences([message], start_token=True, sep_token=True)
num_inputs = len(digits[0])
end_token = tokenizer.get_special_token("end")
#%%
digits = digits.astype(np.int32)
# %%

#%%
with open(engine_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
# %%
input_binding = engine.get_binding_name(0)
output_binding = engine.get_binding_name(1)
# %%
input_shape = engine.get_binding_shape(input_binding)
output_shape = engine.get_binding_shape(output_binding)
# %%
input_type = trt.nptype(engine.get_binding_dtype(input_binding))
output_type = trt.nptype(engine.get_binding_dtype(output_binding))
#%%
token_size = output_shape[-1]
# %%
in_size = num_inputs
out_size = num_inputs * output_shape[-1]
#%%
context = engine.create_execution_context()
# %%
input_buffer = cuda.mem_alloc(max_ctx * np.dtype(input_type).itemsize)
output_buffer = cuda.mem_alloc(max_ctx * output_shape[-1] * np.dtype(output_type).itemsize)
bindings = [int(input_buffer), int(output_buffer)]
#%%
infer_start = time.time()
for _ in range(max_ctx):
    cuda.memcpy_htod(bindings[0], digits)
    context.set_input_shape(name='input', shape=(1, digits.shape[-1]))
    context.execute_v2(bindings=bindings)
    output_data = np.empty(shape=(1, digits.shape[-1], token_size), dtype=np.float32)
    cuda.memcpy_dtoh(output_data, bindings[1])
    pred_token = np.argmax(output_data[:, -1, :], axis=-1)

    if pred_token == end_token:
        break

    digits = np.concatenate((digits, np.array([pred_token], dtype=np.int32)), axis=-1)
infer_stop = time.time()
# %%
print(infer_stop - infer_start)
# %%

#%%
cuda.memcpy_htod(input_buffer, digits)
#%%
context.set_input_shape(name='input', shape=(1, digits.shape[-1]))
# %%
context.execute_v2(bindings=[int(input_buffer), int(output_buffer)])
# %%
output_data = np.empty(digits.shape[-1] * token_size, dtype=np.float32)
# %%
cuda.memcpy_dtoh(output_data, output_buffer)
# %%
output = np.reshape(output_data, (1, digits.shape[-1], -1))
# %%
token = np.argmax(output[:, -1, :])
# %%
digits = np.concatenate((digits, np.array([[token]], dtype=np.int32)), axis=-1)
# %%
digits
# %%
token
#%%
tokenizer.dictionary[token]
# %%
for item in digits[0][num_inputs:]:
    print(tokenizer.dictionary[item])
# %%

# %%

# %%
