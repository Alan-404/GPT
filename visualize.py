from gpt import GPT
import torch
from typing import Callable
from preprocess.data import Tokenizer
import torchviz
from trainer import activation_functions_dict

def draw_model(tokenizer_path: str, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float):
    tokenizer = Tokenizer(tokenizer_path)
    model = GPT(
        len(tokenizer.dictionary), n, d_model, heads, d_ff, activation, dropout_rate, eps
    )

    dummpy_input = torch.randint(low=0, high=len(tokenizer.dictionary), size=(1,15))
    dot = torchviz.make_dot(model(dummpy_input), params = dict(model.named_parameters()))

    dot.render("./assets/gpt_model", format='png')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str)
    # model config
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    args = parser.parse_args()

    if activation_functions_dict[args.activation] is None:
        args.activation = activation_functions_dict['gelu']
    else:
        args.activation = activation_functions_dict[args.activation]

    draw_model(
        tokenizer_path=args.tokenizer_path,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        eps=args.eps
    )