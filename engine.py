#%%
import tensorrt as trt

TRT_LOGGER = trt.Logger()
 
def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine, context
# %%
engine, context = build_engine("./built_models/gpt.onnx")
# %%
builder = trt.Builder(TRT_LOGGER)
# %%
dir(builder)
# %%
with open('./learn.txt', 'w') as file:
    for item in dir(builder):
        file.write(item + "\n")
# %%

# %%
