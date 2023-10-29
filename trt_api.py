from fastapi import FastAPI
from inference import TensorRTEngine
import torch
from preprocessing.tokenizer import Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

tokenizer = Tokenizer("./tokenizers/dictionary.pkl")

engine = TensorRTEngine("./built_models/gpt.trt", token_size=len(tokenizer.dictionary), end_token=tokenizer.get_special_token("end"))
end_token = tokenizer.get_special_token("end")
token_size = len(tokenizer.dictionary)

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

# DTO
class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
def hello(dto: ChatMessage):
    message = dto.message
    logits = tokenizer.text2digit(message, start_token=True, sep_token=True)
    len_token = logits.size
    
    digits = engine(torch.tensor(logits), max_ctx=250)
    digits = digits[0, len_token:]
    text_out = tokenizer.decode(digits)

    return {'response': text_out}


# Start FastAPI App
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)