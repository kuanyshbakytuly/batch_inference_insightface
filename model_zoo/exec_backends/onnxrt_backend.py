import onnxruntime
import cv2
import numpy as np
import logging


class Arcface:
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 swapRB=True,
                 **kwargs):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input_mean = input_mean
        self.input_std = input_std
        self.swapRB = swapRB
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        face_img = np.stack(face_img)

        input_size = tuple(face_img[0].shape[0:2][::-1])
        blob = cv2.dnn.blobFromImages(face_img, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=self.swapRB)

        net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: blob})
        return net_out[0]


class DetectorInfer:

    def __init__(self, model='/models/onnx/centerface/centerface.onnx',
                 output_order=None, **kwargs):

        self.rec_model = onnxruntime.InferenceSession(model)
        logging.info('Detector started')
        self.input = self.rec_model.get_inputs()[0]
        self.input_dtype = self.input.type
        if self.input_dtype == 'tensor(float)':
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.uint8

        self.output_order = output_order
        self.out_shapes = None
        self.input_shape = tuple(self.input.shape)

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up face detection ONNX Runtime engine...")
        if self.output_order is None:
            self.output_order = [e.name for e in self.rec_model.get_outputs()]
        self.out_shapes = [e.shape for e in self.rec_model.get_outputs()]
        self.rec_model.run(self.output_order,
                           {self.rec_model.get_inputs()[0].name: [
                               np.zeros(tuple(self.input.shape[1:]), self.input_dtype)]})

    def run(self, input):
        net_out = self.rec_model.run(self.output_order, {self.input.name: input})
        return net_out
