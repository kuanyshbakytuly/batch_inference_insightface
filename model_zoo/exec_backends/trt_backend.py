import os
import numpy as np
from ..exec_backends.trt_loader import TrtModel

def normalize_image(input, mean=0., std=1., swapRB=False):
    """
    Normalizes the input image using NumPy.

    Args:
        input (np.ndarray): The input image as a NumPy array.
        mean (float, optional): The mean value for normalization. Defaults to 0.
        std (float, optional): The standard deviation for normalization. Defaults to 1.
        swapRB (bool, optional): Whether to swap the red and blue channels. Defaults to True.

    Returns:
        np.ndarray: The normalized image.
    """
    if swapRB:
        input = input[..., ::-1]  # Swaps the last axis (assumed R and B channels)
    normalized_img = input
    # # Move the channels to the second dimension (N, C, H, W)
    if input.ndim == 4:  # Assuming NCHW format
        normalized_img = np.transpose(input, (0, 3, 1, 2))
    elif input.ndim == 3:  # Assuming single image CHW format
         normalized_img = np.transpose(input, (2, 0, 1))
    else:
         raise ValueError("Input must be a 3D or 4D numpy array")

    # Normalize the image
    normalized_img = (normalized_img - mean) / std
    return normalized_img

class Arcface:

    def __init__(self, rec_name: str = '/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 swapRB=False,
                 **kwargs):
        self.rec_model = TrtModel(rec_name)
        self.model_name = os.path.basename(rec_name)
        self.input_mean = input_mean
        self.input_std = input_std

    # warmup
    def prepare(self, **kwargs):
        self.rec_model.build()
        self.out_shapes = self.rec_model.out_shapes
        self.inputs_shape = self.rec_model.inputs_shape
        self.batch_size = self.rec_model.inputs_shape[0][0]

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]
        face_img = np.stack(face_img)

        #infer_shape = _normalize_on_device(face_img, self.stream, self.input_ptr, mean=self.input_mean,std=self.input_std, swapRB=self.swapRB)
        face_img = normalize_image(face_img, self.input_mean, self.input_std)
        embeddings = self.rec_model.run(input=face_img, deflatten=True, as_dict=True)
        self.output_order = ['683']
        embeddings = [embeddings[e] for e in self.output_order]
        return embeddings[0]



class DetectorInfer:

    def __init__(self, model='/models/trt-engines/centerface/centerface.plan',
                 output_order=None, **kwargs):
        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)

    # warmup
    def prepare(self, **kwargs):
        print('Preparing DetectorInfer')
        self.rec_model.build()
        self.out_shapes = self.rec_model.out_shapes
        self.inputs_shape = self.rec_model.inputs_shape
        self.batch_size = self.rec_model.inputs_shape[0][0]



    def run(self, input=None, from_device=False, infer_shape=None, **kwargs):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True, from_device=from_device)
        self.output_order = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
        net_out = [net_out[e] for e in self.output_order]
        return net_out
