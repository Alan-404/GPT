import torch
from trainer import GPTTrainer
from data import Tokenizer
import os

def evaluate_model(test_data_path: str, tokenizer_path: str, checkpoint: str, batch_size: int, device: str):
    tokenizer = Tokenizer(tokenizer_path)

    test_data = tokenizer.get_data(test_data_path)
    test_data = torch.tensor(test_data)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        checkpoint=checkpoint,
        device=device
    )

    trainer.evaluate(test_data, batch_size)

def __check_path(path: str):
    return os.path.exists(path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    assert args.data_path is not None or args.tokenizer_path is not None or args.checkpoint is not None
    assert __check_path(args.data_path) == True and __check_path(args.tokenizer_path) == True and __check_path(args.checkpoint) == True

    evaluate_model(
        test_data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device
    )   