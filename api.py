from fastapi import FastAPI
import onnxruntime as ort
import uvicorn
from preprocessing.tokenizer import Tokenizer
from argparse import ArgumentParser
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import os
import re
import json

# Get Information about Server
parser = ArgumentParser()
parser.add_argument('--host', default='127.0.0.1', help='Host IP to bind to')
parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
parser.add_argument('--model', type=str)
parser.add_argument('--tokenizer', type=str)
parser.add_argument('--max_ctx', type=int, default=201)
parser.add_argument("--device", type=str, default='cpu')

args = parser.parse_args()

assert os.path.exists(args.model) == True and os.path.exists(args.tokenizer) == True

# Config App and APIs
app = FastAPI()

# Cors Config
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Config and Load ONNX Model
providers = ['CPUExecutionProvider']
if args.device.strip().lower() in ['cuda', 'gpu'] and ort.get_device() == 'GPU':
    providers = ['CUDAExecutionProvider'] + providers

session = ort.InferenceSession(args.model, providers=providers)

# Load Tokenizer
tokenizer = Tokenizer(args.tokenizer)
end_token = tokenizer.get_special_token("end")

# DTO
class ChatMessage(BaseModel):
    message: str

def load_json_file(path: str):
    return json.load(open(path, encoding='utf-8'))

# API
@app.post("/chat")
def hello(dto: ChatMessage):
    start_time = time.time()
    request_text_length = len(dto.message.split(" "))

    # Pre-Process Textual Data
    digits = tokenizer.text2digit(dto.message, start_token=True, sep_token=True)

    if digits is None:
        return {'response': "Tôi không biết về điều này, xin lỗi bạn nhé."}
    digits = np.expand_dims(digits, axis=0)
    
    request_token_length = digits.shape[1]
    response_start_index = digits.shape[-1]

    # Generate Tokens Stage
    for _ in range(args.max_ctx):
        output = session.run(
            None,
            {'input': digits}
        )[0]

        pred_token = np.argmax(output[:, -1, :], axis=-1)
        if pred_token == end_token:
            break

        digits = np.concatenate((digits, np.expand_dims(pred_token, axis=0)), axis=-1)
    print(digits)
    digits = digits[0][response_start_index:]
    response_token_length = len(digits)
    
    # Post-Process Data
    response = tokenizer.decode(digits)
    # response_text_length = len(response.split(" "))
    end_time = time.time()

    # Response
    return {'response': response, 
            "time_out": end_time-start_time,
            "request_text_length": request_text_length,
            "request_token_length": request_token_length,
            # 'response_text_length': response_text_length,
            "response_token_length": response_token_length}



# Start FastAPI App
if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)