import tensorrt as trt
import numpy as np
from pycuda import driver as cuda, autoinit # Need to import autoinit to allocate cuda memory
import os
from preprocess import Tokenizer
import time
import re

INPUT_NAME = 'input'
END_TOKEN = "end"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
INPUT_PRECISION = np.int32
OUTPUT_PRECISION = np.float32

def load_engine(engine_path: str):
    if os.path.exists(engine_path) == False:
        print("Not Found Engine Path")
        return None

    
    with open(engine_path, 'rb') as file:
        engine_data = file.read()
    
    # Create Runtime and Engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

    print("OK")
    
    return engine

def allocate_buffer(max_ctx: int, token_size: int):
    input_buffer = cuda.mem_alloc(max_ctx * np.dtype(INPUT_PRECISION).itemsize)
    output_buffer = cuda.mem_alloc(max_ctx * token_size * np.dtype(OUTPUT_PRECISION).itemsize)
    bindings = [int(input_buffer), int(output_buffer)]

    return bindings

def generate_text(context, bindings, digits: np.ndarray, max_ctx: int, end_token, token_size):
    digits = digits.astype(INPUT_PRECISION)
    for _ in range(max_ctx):
        # Copy Data from CPU to GPU Buffer
        cuda.memcpy_htod(bindings[0], digits)
        # Setup Input Shape
        context.set_input_shape(name=INPUT_NAME, shape=(1, digits.shape[-1]))
        # Inference Stage
        context.execute_v2(bindings=bindings)
        # Get Data from GPU Buffer to CPU
        output_data = np.empty(shape=(1, digits.shape[-1], token_size), dtype=OUTPUT_PRECISION)
        cuda.memcpy_dtoh(output_data, bindings[1])
        # Handle Predicted Token
        pred_token = np.argmax(output_data[:, -1, :], axis=-1).astype(INPUT_PRECISION)

        if pred_token[0] == end_token:
            break

        digits = np.concatenate((digits, np.expand_dims(pred_token, axis=0)), axis=-1)
    return digits


def cmd_chat(engine_path: str, tokenizer_path: str, max_ctx: int):
    # Load Tokenizer
    if os.path.exists(tokenizer_path) == False:
        return None
    tokenizer = Tokenizer(tokenizer_path)
    end_token = tokenizer.get_special_token(END_TOKEN)
    
    # Load TensorRT Engine
    load_start_time = time.time()
    engine = load_engine(engine_path)

    if engine is None:
        return None

    token_size = len(tokenizer.dictionary)
    
    # Allocate Buffer Memory in GPU
    bindings = allocate_buffer(max_ctx, token_size)
    
    # Create Engine Context
    context = engine.create_execution_context()
    load_end_time = time.time()

    print(f"Loading Time: {load_end_time - load_start_time}")

    while True:
        # Get Input Message
        message = input("Input Message: ")
        # Pre-process Data
        digits = tokenizer.text_to_sequences([message], start_token=True, sep_token=True)

        message_length = digits.shape[-1]

        try:
            infer_start_time = time.time()
            # Inference
            digits = generate_text(context, bindings, digits, max_ctx, end_token, token_size)
            infer_end_time = time.time()

            # Post-Process
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
                        upper_flag = True
                    elif word == "<circle_dot>":
                        response += "| "
                        upper_flag = True
                    elif word == ".":
                        upper_flag = True
                    continue
                else:
                    if upper_flag:
                        upper_flag = False
                        response += str(word).capitalize()
                    elif upper_all_flag:
                        upper_all_flag = False
                        response += re.sub(tokenizer.word_break_icon, " ", str(word)).upper()
                    else:
                        response += word

            response = re.sub(tokenizer.word_break_icon, " ", response)
            response = re.sub(f"\s{tokenizer.cleaner.puncs}\s", r'\1 ', response)
            response = response[0].upper() + response[1:]
        except Exception as e:
            print(str(e))
            response = "BUG"
            infer_end_time = 0

        
        print(f"Response:\n{response}")
        print(f"Inference Time: {infer_end_time - infer_start_time}")

        exit = input('Do you want to exit? (y/n): ').lower().strip() == 'y'

        if exit:
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--engine_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--max_ctx", type=int, default=250)

    args = parser.parse_args()

    cmd_chat(
        engine_path=args.engine_path,
        tokenizer_path=args.tokenizer_path,
        max_ctx=args.max_ctx
    )

