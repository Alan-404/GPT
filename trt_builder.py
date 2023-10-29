import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorrt as trt
from typing import Union

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