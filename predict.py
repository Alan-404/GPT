import torch
from data import Tokenizer
from trainer import GPTTrainer

def predict_response(tokenizer_path: str, checkpoint: str, device: str):
    tokenizer = Tokenizer(tokenizer_path)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        checkpoint=checkpoint,
        device=device
    )

    while True:
        text_input = input()

        digits = tokenizer.text_to_sequences([text_input], start_token=True, sep_token=True)
        digits = torch.tensor(digits)

        