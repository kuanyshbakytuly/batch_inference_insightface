import os
import numpy as np
import logging
import time
from ..exec_backends.trt_loader import TrtModel

class Arcface:

    def __init__(self, rec_name: str = '/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 swapRB=False,
                 **kwargs):
        self.rec_model = TrtModel(rec_name)
        self.model_name = os.path.basename(rec_name)

    # warmup
    def prepare(self, **kwargs):
        logging.info(f"Warming up Face Recognition ArcFace engine...")
        self.rec_model.build()

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]
        face_img = np.stack(face_img)

        #infer_shape = _normalize_on_device(face_img, self.stream, self.input_ptr, mean=self.input_mean,std=self.input_std, swapRB=self.swapRB)

        embeddings = self.rec_model.run(input=face_img, deflatten=True, as_dict=True)
        self.output_order = ['683']
        embeddings = [embeddings[e] for e in self.output_order]
        return embeddings



class DetectorInfer:

    def __init__(self, model='/models/trt-engines/centerface/centerface.plan',
                 output_order=None, **kwargs):
        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)

    # warmup
    def prepare(self, **kwargs):
        logging.info(f"Warming up Face Detector SCRDF engine...")
        self.rec_model.build()


    def run(self, input=None, from_device=False, infer_shape=None, **kwargs):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True, from_device=from_device)
        self.output_order = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
        net_out = [net_out[e] for e in self.output_order]
        return net_out
