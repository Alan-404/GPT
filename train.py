import torch
from typing import Callable
from trainer import GPTTrainer, activation_functions_dict
from preprocessing.data import Tokenizer, load_data

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
                val_size: float,
                tracking: bool,
                tracking_uri: str,
                experiment_name: str,
                run_id: str,
                run_name: str):
    
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

    trainer.fit(data, 
                epochs=epochs, 
                batch_size=batch_size, 
                mini_batch=mini_batch, 
                val_type=val_type, 
                num_folds=num_folds, 
                val_size=val_size,
                tracking=tracking,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                run_id=run_id,
                run_name=run_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Data Config
    parser.add_argument("--data_path", type=str, help="Path of Pre-processed Dataset.")
    parser.add_argument("--tokenizer_path", type=str, help="Path of trained Tokenizer.")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint you want to load and save.")

    # Model Config
    parser.add_argument("--n", type=int, default=12, help="Number of Decoder Layers")
    parser.add_argument("--d_model", type=int, default=768, help="Number of Word Embedding Dimension.")
    parser.add_argument("--heads", type=int, default=12, help="Number of heads in Multi-head Attention Layer.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Number of hidden dimensions in Position wise Feed Forward Networks.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Probability of set 0.")
    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon in Norm Layer.")
    parser.add_argument("--activation", type=str, default='gelu', help="Activation function between 2 layers in Position wise Feed Forward Networks.")

    # Hyper-params Config
    parser.add_argument("--hyper_path", type=str, default='./model.yml')
    parser.add_argument("--model_name", type=str, default=None)
    
    # Traning Config
    parser.add_argument("--device", type=str, default='cpu', help="Device for Training Stage")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Training Epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    # Validation Config
    parser.add_argument("--val_type", type=str, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--val_size", type=float, default=0.2)

    # Tracking Config
    parser.add_argument("--tracking", type=bool, default=False)
    parser.add_argument("--tracking_uri", type=str, default="/home/alan/Src/Trackings")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    assert args.data_path is not None or args.tokenizer_path is not None

    if activation_functions_dict[args.activation] is None:
        args.activation = activation_functions_dict['gelu']
    else:
        args.activation = activation_functions_dict[args.activation]

    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    if args.model_name is not None:
        import yaml
        from yaml.loader import SafeLoader

        with open(args.hyper_path) as file:
            model_info = yaml.load(file, Loader=SafeLoader)

        params = None
        for item in model_info['models']:
            if item['name'] == args.model_name:
                params = item['hyper_params']
        
        assert params is not None
        args.n = int(params['n'])
        args.d_model = int(params['d_model'])
        args.heads = int(params['heads'])
        args.d_ff = int(params['d_ff'])
        args.activation = activation_functions_dict[str(params['activation'])]
        args.dropout_rate = float(params['dropout_rate'])
        args.eps = float(params['eps'])
    
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
        val_size=args.val_size,
        tracking=args.tracking,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        run_name=args.run_name
    )    