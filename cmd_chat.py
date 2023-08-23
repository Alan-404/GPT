import torch
from preprocessing.data import Tokenizer
from trainer import GPTTrainer
import re
import numpy as np
import onnxruntime as ort
import time

def onnx_response(tokenizer_path: str, checkpoint: str, device: str, max_ctx: int):
    tokenizer = Tokenizer(tokenizer_path)
    end_token = tokenizer.get_special_token("end")
    load_start_time = time.time()
    providers = ['CPUExecutionProvider']
    if device != "cpu" and ort.get_device() == 'GPU':
        providers = ['CUDAExecutionProvider'] + providers

    onnx_session = ort.InferenceSession(checkpoint, providers=providers)
    load_end_time = time.time()
    print(f"Total loading time: {load_end_time - load_start_time}")

    while True:
        text_input = input("Message: ")
        
        infer_start_time = time.time()
        digits = tokenizer.text_to_sequences([text_input], start_token=True, sep_token=True)

        message_length = digits.shape[1]

        infer_start_time = time.time()
        for _ in range(max_ctx):
            output = onnx_session.run(
                None,
                {'input': digits}
            )[0]

            pred_token = np.argmax(output[:, -1, :], axis=-1)
            if pred_token == end_token:
                break

            digits = np.concatenate((digits, np.expand_dims(pred_token, axis=0)), axis=-1)
        infer_end_time = time.time()
        
        response = tokenizer.decode(digits[0][message_length:])
        
        print(f"Response: {response}")
        print(f"Total inference time: {infer_end_time - infer_start_time}")

        exit = input('Do you want to exit? (y/n): ').lower().strip() == 'y'

        if exit:
            break

def jit_response(tokenizer_path: str, checkpoint: str, device: str, max_ctx: int):
    tokenizer = Tokenizer(tokenizer_path)
    end_token = tokenizer.get_special_token("end")

    print(device)

    load_start_time = time.time()
    model = torch.jit.load(checkpoint)
    model.eval()
    model.to(device)
    load_end_time = time.time()

    print(f"Total loading time: {load_end_time - load_start_time}")

    while True:
        text_input = input("Message: ")
        infer_start_time = time.time()
        digits = tokenizer.text_to_sequences([text_input], start_token=True, sep_token=True)
        digits = torch.tensor(digits).to(device)

        message_length = digits.shape[1]
        
        for _ in range(max_ctx):
            output = model(digits)

            _, pred_token = torch.max(output[:, -1, :], dim=-1)

            if pred_token == end_token:
                break

            digits = torch.concatenate((digits, pred_token.unsqueeze(0)), dim=-1)
            print(digits)
        infer_end_time = time.time()

        response = tokenizer.decode(digits[0][message_length:])

        print(f"Response\n: {response}")
        print(f"Total inference time: {infer_end_time - infer_start_time}")

        exit = input('Do you want to exit? (y/n): ').lower().strip() == 'y'

        if exit:
            break


def trainer_repsonse(tokenizer_path: str, checkpoint: str, device: str, max_ctx: int):
    tokenizer = Tokenizer(tokenizer_path)
    
    load_start_time = time.time()
    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        checkpoint=checkpoint,
        device=device
    )
    load_end_time = time.time()
    print(f"Total loading time: {load_end_time - load_start_time}")

    while True:
        text_input = input("Message: ")
        infer_start_time = time.time()
        digits = tokenizer.text_to_sequences([text_input], start_token=True, sep_token=True)
        digits = torch.tensor(digits).to(device)

        message_length = digits.shape[1]
        digits_output = trainer.generate(digits, max_ctx, tokenizer.get_special_token("end"))
        infer_end_time = time.time()
        response = tokenizer.decode(digits_output[0][message_length:])

        print(f"Response: {response}")
        print(f"Total inference time: {infer_end_time - infer_start_time}")

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
    parser.add_argument("--model_type", default='trainer')

    args = parser.parse_args()

    if args.model_type == 'trainer':
        print("Trainer Mode")
        trainer_repsonse(
            tokenizer_path=args.tokenizer_path,
            checkpoint=args.checkpoint,
            device=args.device,
            max_ctx=args.max_ctx
        )
    elif args.model_type == 'onnx':
        print("ONNX Mode")
        onnx_response(
            tokenizer_path=args.tokenizer_path,
            checkpoint=args.checkpoint,
            device=args.device,
            max_ctx=args.max_ctx
        )
    else:
        print("TorchScript Mode")
        jit_response(
            tokenizer_path=args.tokenizer_path,
            checkpoint=args.checkpoint,
            device=args.device,
            max_ctx=args.max_ctx
        )