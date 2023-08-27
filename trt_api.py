from fastapi import FastAPI
import tensorrt as trt
import uvicorn
from pycuda import driver as cuda, autoinit
from preprocessing.data import Tokenizer
from argparse import ArgumentParser
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import os

INPUT_NAME = 'input'
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
INPUT_PRECISION = np.int32
OUTPUT_PRECISION = np.float32

parser = ArgumentParser()
parser.add_argument('--host', default='127.0.0.1', help='Host IP to bind to')
parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
parser.add_argument('--engine', type=str)
parser.add_argument('--tokenizer', type=str)
parser.add_argument('--max_ctx', type=int, default=201)

args = parser.parse_args()

tokenizer = Tokenizer(args.tokenizer)
token_size = len(tokenizer.dictionary)
end_token = tokenizer.get_special_token("end")

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(args.engine, 'rb') as file:
    engine_data = file.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

input_buffer = cuda.mem_alloc(args.max_ctx * np.dtype(INPUT_PRECISION).itemsize)
output_buffer = cuda.mem_alloc(args.max_ctx * np.dtype(OUTPUT_PRECISION).itemsize)
bindings = [int(input_buffer), int(output_buffer)]

context = engine.create_execution_context()

# DTO
class ChatMessage(BaseModel):
    message: str

# API
@app.post("/chat")
def hello(chat_message: ChatMessage):
    start_time = time.time()
    request_text_length = len(chat_message.message.split(" "))

    # Pre-Process Textual Data
    digits = tokenizer.text_to_sequences([chat_message.message], start_token=True, sep_token=True)
    digits = digits.astype(np.int32)
    request_token_length = digits.shape[1]
    response_start_index = digits.shape[-1]

    # Generate Tokens Stage
    for _ in range(args.max_ctx):
        cuda.memcpy_htod(bindings[0], digits)
        context.set_input_shape(name=INPUT_NAME, shape=(1, digits.shape[-1]))
        context.execute_v2(bindings=bindings)
        output_data = np.zeros(shape=(1, digits.shape[-1], token_size), dtype=np.float32)
        cuda.memcpy_dtoh(output_data, bindings[1])
        pred_token = np.argmax(output_data[:, -1, :], axis=-1)
        print(pred_token)

        if pred_token == end_token:
            break

        digits = np.concatenate((digits, np.expand_dims(pred_token.astype(np.int32), axis=0)), axis=-1)

    digital_response = digits[0][response_start_index:]
    response_token_length = len(digital_response)
    
    # Post-Process Data
    response = tokenizer.decode(digital_response)
    
    response_text_length = len(response.split(" "))
    end_time = time.time()

    # Response
    return {'response': response, 
            "time_out": end_time-start_time,
            "request_text_length": request_text_length,
            "request_token_length": request_token_length,
            'response_text_length': response_text_length,
            "response_token_length": response_token_length}



# Start FastAPI App
if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)