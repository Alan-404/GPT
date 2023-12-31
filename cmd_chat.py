import torch
from preprocessing.tokenizer import Tokenizer
from trainer import GPTTrainer, activation_functions_dict
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
        
        words = tokenizer.decode(digits[0][message_length:])
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
                continue
            else:
                if upper_flag:
                    upper_flag = False
                    response += str(word).capitalize()
                elif upper_all_flag:
                    upper_all_flag = False
                    response += str(word).upper()
                else:
                    response += word

        response = re.sub(tokenizer.word_break_icon, " ", response).capitalize()
        
        print(f"Response:\n{response}")
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

        infer_end_time = time.time()

        response = tokenizer.decode(digits[0][message_length:])

        print(f"Response\n: {response}")
        print(f"Total inference time: {infer_end_time - infer_start_time}")

        exit = input('Do you want to exit? (y/n): ').lower().strip() == 'y'

        if exit:
            break


def trainer_repsonse(
        n:int,
        d_model: int,
        heads: int,
        d_ff: int,
        eps: float,
        activation,
        dropout_rate: float,
        tokenizer_path: str, checkpoint: str, device: str, max_ctx: int
    ):
    tokenizer = Tokenizer(tokenizer_path)
    
    load_start_time = time.time()
    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        n=n,
        heads=heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        eps=eps,
        activation=activation,

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

    # Model Hyper-parameters
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device",type=str, default='cpu')
    parser.add_argument("--max_ctx", type=int, default=250)
    parser.add_argument("--model_type", default='trainer')

    args = parser.parse_args()

    if args.model_type == 'trainer':
        print("Trainer Mode")
        trainer_repsonse(
            n=args.n,
            d_model=args.d_model,
            heads=args.heads,
            d_ff=args.d_ff,
            eps=args.eps,
            activation=activation_functions_dict[args.activation],
            dropout_rate=args.dropout_rate,
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