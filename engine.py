import tensorrt as trt
import numpy as np
from pycuda import driver as cuda, autoinit
import os
from preprocess import Tokenizer
import time
import re

INPUT_NAME = 'input'

INPUT_PRECISION = np.int32
OUTPUT_PRECISION = np.float32

def cmd_chat(engine_path: str, tokenizer_path: str, max_ctx: int):
    if os.path.exists(tokenizer_path) == False:
        return None
    tokenizer = Tokenizer(tokenizer_path)
    end_token = tokenizer.get_special_token("end")
    if os.path.exists(engine_path) == False:
        print("Not Found Engine Path")
        return None

    load_start_time = time.time()
    with open(engine_path, 'rb') as file:
        engine_data = file.read()
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        return None


    token_size = len(tokenizer.dictionary)
    
    input_buffer = cuda.mem_alloc(max_ctx * np.dtype(INPUT_PRECISION).itemsize)
    output_buffer = cuda.mem_alloc(max_ctx * token_size * np.dtype(OUTPUT_PRECISION).itemsize)
    bindings = [int(input_buffer), int(output_buffer)]
    
    context = engine.create_execution_context()
    load_end_time = time.time()

    print(f"Loading Time: {load_end_time - load_start_time}")

    while True:
        message = input("Input Message: ")
        digits = tokenizer.text_to_sequences([message], start_token=True, sep_token=True)
        digits = digits.astype(np.int32)
        message_length = digits.shape[-1]

        try:
            infer_start_time = time.time()
            for _ in range(max_ctx):
                cuda.memcpy_htod(bindings[0], digits)
                context.set_input_shape(name=INPUT_NAME, shape=(1, digits.shape[-1]))
                context.execute_v2(bindings=bindings)
                output_data = np.zeros(shape=(1, digits.shape[-1], token_size), dtype=np.float32)
                cuda.memcpy_dtoh(output_data, bindings[1])
                pred_token = np.argmax(output_data[:, -1, :], axis=-1)
                if pred_token == end_token:
                    break

                digits = np.concatenate((digits, np.array([pred_token], dtype=np.int32)), axis=-1)
            infer_end_time = time.time()
            response = tokenizer.decode(digits[0][message_length:])
            response = re.sub("<new_line>", "\n", response)
            response = response.title()
        
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

