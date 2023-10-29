import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorrt as trt
from pycuda import driver as cuda, autoinit
from typing import Union
import numpy as np

class TensorRTBuilder:
    def __init__(self, min_shapes: Union[list, tuple], opt_shapes:Union[list, tuple], max_shapes: Union[list, tuple], fp16: bool = False, int8: bool = False, gpu_fallback: bool = False, input_name: str = 'input') -> None:
        assert fp16 != int8
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        self.min_shapes = min_shapes
        self.opt_shapes = opt_shapes
        self.max_shapes = max_shapes

        self.input_name = input_name
        self.gpu_fallback = gpu_fallback

        self.fp16 = fp16
        self.int8 = int8

    def build(self, onnx_path: str, engine_path: str):
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        config = builder.create_builder_config()

        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(onnx_path, 'rb') as model:
            parser.parse(model.read(), path="/onnx")

        profile = builder.create_optimization_profile()
        profile.set_shape(self.input_name, self.min_shapes, self.opt_shapes, self.max_shapes)

        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)

        if self.gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        config.add_optimization_profile(profile)

        serialized_network = builder.build_serialized_network(network, config)
        with trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_network)

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

class TensorRTEngine:
    def __init__(self, engine_path: str, token_size: int, end_token: int, input_name: str = 'input') -> None:
        self.engine_path = engine_path
        self.INPUT_NAME = input_name
        self.END_TOKEN = end_token
        self.TOKEN_SIZE = token_size

        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.INPUT_PRECISION = np.int32
        self.OUTPUT_PRECISION = np.float32

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()


    def load_engine(self, engine_path: str):
        if os.path.exists(self.engine_path) == False:
            print("Not Found Engine Path")
            return None

        
        with open(engine_path, 'rb') as file:
            engine_data = file.read()
        
        # Create Runtime and Engine
        runtime = trt.Runtime(self.TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        return engine

    def __call__(self, logits: np.ndarray, max_ctx: int):

        logits = np.expand_dims(logits, axis=0).astype(self.INPUT_PRECISION)

        for _ in range(max_ctx):
            # Allocate buffer
            input_buffer = cuda.mem_alloc(logits.size * np.dtype(self.INPUT_PRECISION).itemsize)
            output_buffer = cuda.mem_alloc(logits.size * self.TOKEN_SIZE * np.dtype(self.OUTPUT_PRECISION).itemsize)
            bindings = [int(input_buffer), int(output_buffer)]

            # Host data from CPU to GPU
            cuda.memcpy_htod(bindings[0], logits)
            self.context.set_input_shape(name=self.INPUT_NAME, shape=(1, logits.shape[-1]))

            # Execute
            self.context.execute_v2(bindings=bindings)
            
            # Get Data from GPU to CPU
            preds = np.empty(shape=(1, logits.shape[-1], self.TOKEN_SIZE), dtype=self.OUTPUT_PRECISION)

            cuda.memcpy_dtoh(preds, bindings[1])

            pred_token = np.argmax(preds[:, -1, :], axis=-1).astype(self.INPUT_PRECISION)

            if pred_token[0] == self.END_TOKEN:
                break

            logits = np.concatenate((logits, np.expand_dims(pred_token, axis=0)), axis=-1)
        return logits

    def __del__(self):
        del self.TRT_LOGGER
        del self.engine
        del self.context     