import torch
from trainer import GPTTrainer, activation_functions_dict
from preprocessing.data import Tokenizer
from typing import Callable
import os

def evaluate_model(test_data_path: str, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float, tokenizer_path: str, checkpoint: str, batch_size: int, device: str):
    tokenizer = Tokenizer(tokenizer_path)

    test_data = tokenizer.get_data(test_data_path)
    test_data = torch.tensor(test_data)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        n=n,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        activation=activation,
        dropout_rate=dropout_rate,
        eps=eps,
        checkpoint=checkpoint,
        device=device
    )

    trainer.evaluate(test_data, batch_size)

def __check_path(path: str):
    return os.path.exists(path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Data Config
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    # Model Config
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    # Validate Config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    assert args.data_path is not None or args.tokenizer_path is not None or args.checkpoint is not None
    assert __check_path(args.data_path) == True and __check_path(args.tokenizer_path) == True and __check_path(args.checkpoint) == True

    args.activation = activation_functions_dict[args.activation]

    evaluate_model(
        test_data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        eps=args.eps
    )