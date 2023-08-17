import torch
from typing import Callable
from trainer import GPTTrainer, activation_functions_dict
from data import Tokenizer, load_data

def train_model(data_path: str,
                tokenizer_path: str,
                n: int,
                d_model: int,
                heads: int,
                d_ff: int,
                dropout_rate: float,
                eps: float,
                activation: Callable[[torch.Tensor], torch.Tensor],
                device: str,
                checkpoint: str,
                epochs: int,
                batch_size: int,
                mini_batch: int,
                learning_rate: float,
                val_type: str,
                num_folds: int,
                val_size: float):
    
    tokenizer = Tokenizer(tokenizer_path)
    
    data = load_data(data_path)
    assert data is not None

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        n=n,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        eps=eps,
        activation=activation,
        device=device,
        checkpoint=checkpoint,
        learning_rate=learning_rate
    )

    data = torch.tensor(data)

    trainer.fit(data, epochs=epochs, batch_size=batch_size, mini_batch=mini_batch, val_type=val_type, num_folds=num_folds, val_size=val_size)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    # model config
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')
    
    # training config
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    # validation config
    parser.add_argument("--val_type", type=str, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--val_size", type=float, default=0.2)


    args = parser.parse_args()


    if args.data_path is None or args.tokenizer_path is None:
        raise Exception("Data Path and Tokenizer path is NOT NONE")

    if activation_functions_dict[args.activation] is None:
        args.activation = activation_functions_dict['gelu']
    else:
        args.activation = activation_functions_dict[args.activation]

    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    train_model(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        dropout_rate=args.dropout_rate,
        activation=args.activation,
        eps=args.eps,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        learning_rate=args.learning_rate,
        val_type=args.val_type,
        num_folds=args.num_folds,
        val_size=args.val_size
    )    