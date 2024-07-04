import tensorrt as trt
import sys
from typing import Union

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _build_engine_onnx(input_onnx: Union[str, bytes], force_fp16: bool = False, max_batch_size: int = 1,
                       max_workspace: int = 1024, set_shape: str = 'model_engine_minminmax'):
    """
    Builds TensorRT engine from provided ONNX file

    :param input_onnx: serialized ONNX model.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :param max_workspace: Maximum builder workspace in MB.
    :return: TensorRT engine
    """

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        has_fp16 = builder.platform_has_fast_fp16
        if force_fp16 or has_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        max_workspace_size = 1 << 30

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        if not parser.parse(input_onnx):
            print('ERROR: Failed to parse the ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

        profile = builder.create_optimization_profile()
        # Get input name and shape for building optimization profile
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        for input in inputs:
            print(f"Model {input.name} shape: {input.shape} {input.dtype}")
            print('-----------------------')
        for output in outputs:
            print(f"Model {output.name} shape: {output.shape} {output.dtype}")
            print('-----------------------')

        input = network.get_input(0)
        inp_shape = list(input.shape)
        if max_batch_size > 1:
            # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles
            profile = builder.create_optimization_profile()
            min_shape = [1] + inp_shape[-3:]
            opt_shape = [int(max_batch_size/2)] + inp_shape[-3:]
            max_shape = [max_batch_size] + inp_shape[-3:]
            for input in inputs:
                print(f"Model {input.name} shape: {max_shape, max_shape, max_shape}")
                profile.set_shape(input.name, max_shape, max_shape, max_shape)

            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config=config)
        if engine is None:
            print('Engine is None')
            return None
        return engine


def check_fp16():
    builder = trt.Builder(TRT_LOGGER)
    has_fp16 = builder.platform_has_fast_fp16
    return has_fp16


def convert_onnx(input_onnx: Union[str, bytes], engine_file_path: str, force_fp16: bool = False,
                 max_batch_size: int = 1, set_shape: str = 'model_engine_minminmax'):
    '''
    Creates TensorRT engine and serializes it to disk
    :param input_onnx: Path to ONNX file on disk or serialized ONNX model.
    :param engine_file_path: Path where TensorRT engine should be saved.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :return: None
    '''

    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
            print('Read')
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx

    engine = _build_engine_onnx(input_onnx=onnx_obj,
                                force_fp16=force_fp16, max_batch_size=max_batch_size, set_shape=set_shape)
    assert not isinstance(engine, type(None))

    with open(engine_file_path, "wb") as f:
        f.write(engine)
