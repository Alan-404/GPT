import torch
from data import Tokenizer
from trainer import GPTTrainer
import re

def predict_response(tokenizer_path: str, checkpoint: str, device: str, max_ctx: int):
    tokenizer = Tokenizer(tokenizer_path)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        checkpoint=checkpoint,
        device=device
    )

    while True:
        text_input = input("Message: ")

        digits = tokenizer.text_to_sequences([text_input], start_token=True, sep_token=True)
        digits = torch.tensor(digits).to(device)

        message_length = digits.shape[1]

        digits_output = trainer.generate(digits, max_ctx, tokenizer.get_special_token("end"))

        response_tokens = []

        for item in digits_output[0][message_length:]:
            response_tokens.append(tokenizer.dictionary[item.item()])

        response = "".join(response_tokens)
        response = re.sub('</w>', " ", response)

        print(f"Response: {response}")

        exit = input('Do you want to exit? (y/n): ').lower().strip() == 'y'

        if exit:
            break

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device",type=str, default='cpu')
    parser.add_argument("--max_ctx", type=int, default=201)

    args = parser.parse_args()

    predict_response(
        tokenizer_path=args.tokenizer_path,
        checkpoint=args.checkpoint,
        device=args.device,
        max_ctx=args.max_ctx
    )
        