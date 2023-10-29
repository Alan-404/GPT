from trainer import GPTTrainer, GPTDataset, activation_functions_dict
from preprocessing.tokenizer import Tokenizer
import warnings
import os

warnings.filterwarnings('ignore')

def train_model(
        manifest_path: str,
        tokenizer_path: str,
        n: int,
        d_model: int,
        heads: int,
        activation: str,
        dropout_rate: float,
        eps: float,
        device: str,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        checkpoint: str,
        tracking_config_path: str,
        val_path: str,
        validation_config_path: str
):
    assert os.path.exists(tokenizer_path) and os.path.exists(manifest_path)

    tokenizer = Tokenizer(tokenizer_path)

    train_dataset = GPTDataset(manifest_path, tokenizer=tokenizer)

    val_dataset = None
    if val_path is not None:
        if os.path.exists(val_path) == True:
            val_dataset = GPTDataset(val_path, tokenizer=tokenizer)

    trainer = GPTTrainer(
        token_size=len(tokenizer.dictionary),
        n=n,
        d_model=d_model,
        heads=heads,
        activation=activation_functions_dict[activation],
        dropout_rate=dropout_rate,
        eps=eps,
        device=device,
        init_learning_rate=learning_rate,
        checkpoint=checkpoint
    )

    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        tracking_config_path=tracking_config_path,
        validation_config_path=validation_config_path
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Pre-config
    parser.add_argument("--manifest_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--val_path", type=str, default=None)

    # Model Hyper-parameters
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    # Training
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="./gpt.pt")


    # Config
    parser.add_argument("--tracking_config_path", type=str, default=None)
    parser.add_argument("--validation_config_path", type=str, default=None)


    # Main Handle
    args = parser.parse_args()


    train_model(
        manifest_path=args.manifest_path,
        tokenizer_path=args.tokenizer_path,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        eps=args.eps,
        device=args.device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        tracking_config_path=args.tracking_config_path,
        validation_config_path=args.validation_config_path,
        val_path=args.val_path
    )

