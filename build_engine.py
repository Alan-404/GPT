import os
from gpt import GPT
from preprocessing.tokenizer import Tokenizer
from trainer import activation_functions_dict
from trt_builder import TensorRTBuilder
import torch

def __build_onnx(model: torch.nn.Module, save_path: str, jit_path: str, device: str = 'cpu'):
    dumpy_input = torch.tensor([[1,2,3,4,5]]).to(device)
    model.to(device)

    model.eval()

    torch.onnx.export(model, dumpy_input, save_path, input_names=["input"], output_names=['output'], dynamic_axes={"input": {0: 'batch_size', 1: "n_ctx"}, "output": {0: "batch_size", 1: "n_ctx"}})
    
    torch.jit.save(torch.jit.trace(model, example_inputs=dumpy_input), jit_path)

def build_engine(onnx_path: str, engine_path: str, min_shape: int, opt_shape: int, max_shape: int, fp16: bool, int8: bool, gpu_fallback: bool):
    builder = TensorRTBuilder(
        min_shapes=min_shape,
        opt_shapes=opt_shape,
        max_shapes=max_shape,
        fp16=fp16,
        int8=int8,
        gpu_fallback=gpu_fallback,
        input_name='input'
    )

    builder.build(
        onnx_path=onnx_path,
        engine_path=engine_path
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str)

    # Model Hyper-parameters
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("--min_shape", nargs="+", type=int)
    parser.add_argument("--opt_shape", nargs="+", type=int, default=None)
    parser.add_argument("--max_shape", nargs="+", type=int)

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--int8", type=bool, default=False)
    parser.add_argument("--gpu_fallback", type=bool, default=False)

    args = parser.parse_args()

    assert args.tokenizer_path is not None and args.checkpoint is not None

    args.min_shape = tuple(args.min_shape)
    args.max_shape = tuple(args.max_shape)

    print(args.min_shape)
    print(args.max_shape)

    if args.opt_shape is None:
        opt_batch_size = int((args.min_shape[0] + args.max_shape[0])/2)
        opt_ctx = int((args.min_shape[1] + args.max_shape[1])/2)
        args.opt_shape = (opt_batch_size, opt_ctx)
    else:
        args.opt_shape = tuple(args.opt_shape)

    if os.path.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)

    tokenizer = Tokenizer(pretrained=args.tokenizer_path)

    model = GPT(
        token_size=len(tokenizer.dictionary),
        n=args.n,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        activation=activation_functions_dict[args.activation],
        dropout_rate=args.dropout_rate,
        eps=args.eps
    )

    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    onnx_path = f"{args.save_folder}/gpt.onnx"
    jit_path = f"{args.save_folder}/gpt.jit"
    engine_path = f"{args.save_folder}/gpt.trt"

    __build_onnx(model, save_path=onnx_path, jit_path=jit_path, device=args.device)

    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        fp16=args.fp16,
        int8=args.int8,
        gpu_fallback=args.gpu_fallback
    )

    print(f"Files is saved at {args.save_folder}")

    

    