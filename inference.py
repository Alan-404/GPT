import os
import tensorrt as trt
import torch
import numpy as np

class TensorRTEngine:
    def __init__(self, engine_path :str, token_size: int, end_token: int, input_name: str = 'input', device: str = 'cuda') -> None:
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

        self.engine_path = engine_path

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.INPUT_PRECISION = torch.int32
        self.OUTPUT_PRECISION = torch.float32

        self.input_name = input_name
        self.token_size = token_size
        self.end_token = end_token
        self.device = device

    def load_engine(self, engine_path: str):
        if os.path.exists(self.engine_path) == False:
            print("Not Found Engine Path")
            return None

        
        with open(engine_path, 'rb') as file:
            engine_data = file.read()
        
        # Create Runtime and Engine
        engine = trt.Runtime(self.TRT_LOGGER).deserialize_cuda_engine(engine_data)
        
        return engine
    
    def __call__(self, digits: torch.Tensor, max_ctx: int = 250):
        digits = digits.unsqueeze(0).to(self.device).type(self.INPUT_PRECISION)

        for _ in range(max_ctx):

            output_data = torch.empty(size=(digits.size(0), digits.size(1), self.token_size), device=self.device, dtype=self.OUTPUT_PRECISION)

            bindings = [int(digits.data_ptr()), int(output_data.data_ptr())]
            self.context.set_input_shape(name=self.input_name, shape=(digits.size(0), digits.size(1)))


            self.context.execute_v2(bindings)

            _, token = torch.max(output_data[:, -1, :], dim=-1)

            if token == self.end_token:
                break

            digits = torch.concat([digits, token.unsqueeze(0)], dim=-1).type(self.INPUT_PRECISION)
        
        return digits.cpu().numpy()  

    def __del__(self):
        del self.TRT_LOGGER
        del self.engine
        del self.context 