import os
import json
from typing import List
import onnx

from .face_detectors import *
from .face_processors import *

# from ..converters.insight2onnx import convert_insight_model
from ..converters.reshape_onnx import reshape
from ..utils.helpers import prepare_folders
from ..utils.helpers import check_hash
from ..configs import Configs

# Since TensorRT, TritonClient and PyCUDA are optional dependencies it might be not available

from .exec_backends import trt_backend
#from .exec_backends import triton_backend as triton_backend
from ..converters.onnx_to_trt import convert_onnx, check_fp16


# Map function names to corresponding functions
func_map = {
    'scrfd': scrfd,
    'scrfd_v2': scrfd_v2,
    'arcface_mxnet': arcface_mxnet,
    'arcface_torch': arcface_torch,
}


def sniff_output_order(model_path, save_dir):
    outputs_file = os.path.join(save_dir, 'output_order.json')
    if not os.path.exists(outputs_file):
        model = onnx.load(model_path)
        output = [node.name for node in model.graph.output]
        with open(outputs_file, mode='w') as fl:
            fl.write(json.dumps(output))
    else:
        output = read_outputs_order(save_dir)
    return output


def read_outputs_order(trt_dir):
    outputs = None
    outputs_file = os.path.join(trt_dir, 'output_order.json')
    if os.path.exists(outputs_file):
        with open(outputs_file, mode='r') as fl:
            outputs = json.loads(fl.read())
    return outputs


def prepare_backend(model_name, backend_name, im_size: List[int] = None,
                    max_batch_size: int = 1,
                    force_fp16: bool = False,
                    download_model: bool = True,
                    config: Configs = None):
    """
    Check if ONNX, MXNet and TensorRT models exist and download/create them otherwise.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param download_model: Download MXNet or ONNX model if it not exist.
    :param config:  Configs class instance
    :return: ONNX model serialized to string, or path to TensorRT engine
    """

    prepare_folders([config.onnx_models_dir, config.trt_engines_dir])
    reshape_allowed = config.models[model_name].get('reshape')
    shape = config.get_shape(model_name)

    if reshape_allowed is True and im_size is not None:
        shape = (1, 3) + tuple(im_size)[::-1]

    onnx_dir, onnx_path = config.build_model_paths(model_name, 'onnx')
    trt_dir, trt_path = config.build_model_paths(model_name, 'engine')

    onnx_exists = os.path.exists(onnx_path)
    onnx_hash = config.models[model_name].get('md5')
    trt_rebuild_required = False
    if onnx_exists and onnx_hash:
        hashes_match = check_hash(onnx_path, onnx_hash, algo='md5')
        if not hashes_match:
            onnx_exists = False
            trt_rebuild_required = True

    if backend_name == 'triton':
        return model_name

    if backend_name == 'onnx':
        model = onnx.load(onnx_path)
        if reshape_allowed is True:
            model = reshape(model, h=im_size[1], w=im_size[0])
        return model.SerializeToString()

    if backend_name == "trt":
        has_fp16 = check_fp16()

        if reshape_allowed is True:
            trt_path = trt_path.replace('.engine', f'_{shape[3]}_{shape[2]}.engine')
        if max_batch_size > 1:
            trt_path = trt_path.replace('.engine', f'_batch{max_batch_size}.engine')
        if force_fp16 or has_fp16:
            trt_path = trt_path.replace('.engine', '_fp16.engine')

        prepare_folders([trt_dir])

        if not config.get_outputs_order(model_name):
            sniff_output_order(onnx_path, trt_dir)

        if not os.path.exists(trt_path) or trt_rebuild_required:

            if reshape_allowed is True or max_batch_size != 1:
                model = onnx.load(onnx_path)
                onnx_batch_size = 1
                if max_batch_size != 1:
                    onnx_batch_size = -1
                reshaped = reshape(model, n=onnx_batch_size, h=shape[2], w=shape[3])
                temp_onnx_model = reshaped.SerializeToString()

            else:
                temp_onnx_model = onnx_path

            convert_onnx(temp_onnx_model,
                         engine_file_path=trt_path,
                         max_batch_size=max_batch_size,
                         force_fp16=force_fp16)
        return trt_path


def get_model(model_name: str, backend_name: str, im_size: List[int] = None, max_batch_size: int = 1,
              force_fp16: bool = False,
              root_dir: str = "/models", download_model: bool = True, triton_uri=None, device_id: str = '0', **kwargs):
    """
    Returns inference backend instance with loaded model.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, mxnet, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param root_dir: Root directory where models will be stored.
    :param download_model: Download MXNet or ONNX model. Might be disabled if TRT model was already created.
    :param kwargs: Placeholder.
    :return: Inference backend with loaded model.
    """

    config = Configs(models_dir=root_dir)

    backends = {
        #'onnx': onnx_backend,
        'trt': trt_backend,
        'mxnet': 'mxnet',
        #'triton': triton_backend
    }

    backend = backends[backend_name]

    model_path = prepare_backend(model_name, backend_name, im_size=im_size, max_batch_size=max_batch_size,
                                 config=config, force_fp16=force_fp16,
                                 download_model=download_model)

    outputs = config.get_outputs_order(model_name)
    if not outputs and backend_name == 'trt':
        trt_dir, trt_path = config.build_model_paths(model_name, 'engine')
        outputs = read_outputs_order(trt_dir)

    func = func_map[config.models[model_name].get('function')]
    model = func(model_path=model_path, backend=backend, device_id=device_id, outputs=outputs, triton_uri=triton_uri)
    return model
