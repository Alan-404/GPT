from trainer import GPTTrainer, activation_functions_dict
from gpt import GPT
from preprocessing.data import Tokenizer
import torch
from typing import Callable
import os

def build_model(checkpoint: str, tokenizer_path: str, build_path: str, device: str, model_type: str, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float):
    # Check Config Paths
    assert os.path.exists(checkpoint) == True and os.path.exists(tokenizer_path) == True
    
    # Handle Device
    if device != 'cpu' and torch.cuda.is_available() == False:
        device = 'cpu'
    
    # Load Tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    model = GPT(
        token_size=len(tokenizer.dictionary),
        n=n,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        activation=activation,
        dropout_rate=dropout_rate,
        eps=eps
    )

    model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    model.eval()

    model.to(device)

    # Init input of Model
    dummpy_input = torch.randint(low=0, high=len(tokenizer.dictionary), size=(1, 20))

    if model_type == 'onnx':
        # Export model to ONNX
        torch.onnx.export(model, dummpy_input.to(device), build_path, input_names=["input"], output_names=['output'], dynamic_axes={"input": {0: 'batch_size', 1: "n_ctx"}, "output": {0: "batch_size", 1: "n_ctx"}})
    else:
        # Export model to JIT
        traced_model = torch.jit.trace(model, example_inputs=dummpy_input.to(device))
        torch.jit.save(traced_model, build_path)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Model Config
    parser.add_argument("--n", type=int, default=12, help="Number of Decoder Layers")
    parser.add_argument("--d_model", type=int, default=768, help="Number of Word Embedding Dimension.")
    parser.add_argument("--heads", type=int, default=12, help="Number of heads in Multi-head Attention Layer.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Number of hidden dimensions in Position wise Feed Forward Networks.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Probability of set 0.")
    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon in Norm Layer.")
    parser.add_argument("--activation", type=str, default='gelu', help="Activation function between 2 layers in Position wise Feed Forward Networks.")

    #
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--build_path", type=str)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_type", type=str, default='onnx')

    args = parser.parse_args()

    build_model(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        build_path=args.build_path,
        device=args.device,
        model_type=args.model_type,
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        activation=activation_functions_dict[args.activation],
        dropout_rate=args.dropout_rate,
        eps=args.eps
    )