from fastapi import FastAPI
from pydantic import BaseModel
import tensorrt as trt
import numpy as np
from preprocessing.data import Tokenizer
import os
import pycuda.driver as cuda
import pycuda.autoinit
import re
import uvicorn

app = FastAPI()

INPUT_NAME = 'input'
END_TOKEN = "end"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
INPUT_PRECISION = np.int32
OUTPUT_PRECISION = np.float32
tokenizer = Tokenizer("./tokenizer/v1/dictionary.pkl")
max_ctx = 250
end_token = tokenizer.get_special_token("end")
token_size = len(tokenizer.dictionary)
engine_path = './engines/v1/gpt.trt'

assert os.path.exists(engine_path)

cuda.init 
device = cuda.Device(0) 
ctx = device.make_context() 

with open(engine_path, 'rb') as file:
    engine_data = file.read()
    
# Create Runtime and Engine
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

input_buffer = cuda.mem_alloc(max_ctx * np.dtype(INPUT_PRECISION).itemsize)
output_buffer = cuda.mem_alloc(max_ctx * token_size * np.dtype(OUTPUT_PRECISION).itemsize)
bindings = [int(input_buffer), int(output_buffer)]

context = engine.create_execution_context()

ctx.pop()

def generate(digits: np.ndarray):
    digits = digits.astype(INPUT_PRECISION)
    for _ in range(max_ctx):
        # Copy Data from CPU to GPU Buffer
        cuda.memcpy_htod(bindings[0], digits)
        # Setup Input Shape
        context.set_input_shape(name=INPUT_NAME, shape=(1, digits.shape[-1]))
        # Inference Stage
        context.execute_v2(bindings=bindings)
        # Get Data from GPU Buffer to CPU
        output_data = np.empty(shape=(1, digits.shape[-1], token_size), dtype=OUTPUT_PRECISION)
        cuda.memcpy_dtoh(output_data, bindings[1])
        # Handle Predicted Token
        pred_token = np.argmax(output_data[:, -1, :], axis=-1).astype(INPUT_PRECISION)

        if pred_token[0] == end_token:
            break

        digits = np.concatenate((digits, np.expand_dims(pred_token, axis=0)), axis=-1)
    return digits

def post_process(digits):
    words = tokenizer.decode(digits)
    response = ""
            
    upper_flag = False
    upper_all_flag = False
    for word in words:
        if word in tokenizer.special_tokens:
            if word == "<upper>":
                upper_flag = True
            elif word == "<upper_all>":
                upper_all_flag = True
            elif word == "<new_line>":
                response += "\n"
                upper_flag = True
            elif word == "<circle_dot>":
                response += "| "
                upper_flag = True
            elif word == ".":
                upper_flag = True
            continue
        else:
            if upper_flag:
                upper_flag = False
                response += str(word).capitalize()
            elif upper_all_flag:
                upper_all_flag = False
                response += re.sub(tokenizer.word_break_icon, " ", str(word)).upper()
            else:
                response += word

    response = re.sub(tokenizer.word_break_icon, " ", response)
    response = re.sub(f"\s{tokenizer.cleaner.puncs}\s", r'\1 ', response)
    response = response[0].upper() + response[1:]

    return response

class MessageDTO(BaseModel):
    message: str

@app.post("/chat")
def chatbot(dto: MessageDTO):
    try:
        digits = tokenizer.text_to_sequences([dto.message], start_token=True, sep_token=True)
        message_length = digits.shape[-1]

        digits = generate(digits)
        response = post_process(digits[0][message_length:])

        return {"response": response}

    except Exception as e:
        print(str(e))
        return {'response': "INTERNAL ERROR SERVER"}
    

if __name__ == '__main__':
    uvicorn.run(app)