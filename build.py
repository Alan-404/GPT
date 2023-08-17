from trainer import GPTTrainer
from data import Tokenizer
import torch
import os


def build_model(checkpoint: str, tokenizer_path: str, build_path: str, device: str):
    if os.path.exists(checkpoint) == False or os.path.exists(tokenizer_path) == False:
        return
    if device != 'cpu' and torch.cuda.is_available() == False:
        device = 'cpu'
    
    tokenizer = Tokenizer(tokenizer_path)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        device=device,
        checkpoint=checkpoint
    )
    trainer.model.eval()

    dummpy_input = torch.randint(low=0, high=len(tokenizer.dictionary), size=(1, 20))

    torch.onnx.export(trainer.model, dummpy_input.to(device), build_path, input_names=["input"], output_names=['output'], dynamic_axes={"input": {0: 'batch_size', 1: "n_ctx"}, "output": {0: "batch_size", 1: "n_ctx"}})

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--build_path", type=str)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    build_model(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        build_path=args.build_path,
        device=args.device
    )