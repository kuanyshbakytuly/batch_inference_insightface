import collections
import logging
import time
from functools import partial
from typing import List

import cv2
import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
import pickle
import base64
import os

from .dataimage import resize_image
from .model_zoo.getter import get_model
from .utils import fast_face_align as face_align
from .utils.cosine import cosine_sim
from .utils.helpers import to_chunks, validate_max_size
from numpy.linalg import norm

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

Face = collections.namedtuple("Face", ['bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm',
                                       'normed_embedding', 'facedata', 'scale', 'num_det', 'mask', 'mask_probs'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)


def serialize_face(_face_dict: dict, return_face_data: bool, return_landmarks: bool = False):
    if _face_dict.get('norm'):
        _face_dict.update(vec=_face_dict['vec'].tolist(),
                          norm=float(_face_dict['norm']))
    # Warkaround for embed_only flag
    if _face_dict.get('prob'):
        _face_dict.update(prob=float(_face_dict['prob']),
                          bbox=_face_dict['bbox'].astype(int).tolist(),
                          size=int(_face_dict['bbox'][2] - _face_dict['bbox'][0]))

    if return_landmarks:
        _face_dict['landmarks'] = _face_dict['landmarks'].astype(int).tolist()
    else:
        _face_dict.pop('landmarks', None)

    if return_face_data:
        _face_dict['facedata'] = base64.b64encode(cv2.imencode('.jpg', _face_dict['facedata'])[1].tostring()).decode(
            'ascii')
    else:
        _face_dict.pop('facedata', None)

    return _face_dict


# Wrapper for insightface detection model
class Detector:
    def __init__(self, det_name: str = 'retinaface_r50_v1', max_size=None,
                 backend_name: str = 'trt', force_fp16: bool = False, triton_uri=None, max_batch_size: int = 1,
                 root_dir='/models'):
        if max_size is None:
            max_size = [640, 480]

        self.retina = get_model(det_name, backend_name=backend_name, force_fp16=force_fp16, im_size=max_size,
                                root_dir=root_dir, download_model=False, triton_uri=triton_uri,
                                max_batch_size=max_batch_size)

        self.retina.prepare(nms=0.35)

    def detect(self, data, threshold=0.3):
        bboxes, landmarks = self.retina.detect(data, threshold=threshold)

        boxes = [e[:, 0:4] for e in bboxes]
        probs = [e[:, 4] for e in bboxes]

        return boxes, probs, landmarks


# Translate bboxes and landmarks from resized to original image size
def reproject_points(dets, scale: float):
    if scale != 1.0:
        dets = dets / scale
    return dets


class FaceAnalysis:
    def __init__(self,
                 det_name: str = 'scrfd_2.5g_gnkps',
                 rec_name: str = 'w600k_r50',
                 max_size=[640, 640],
                 max_batch_size: int = 1,
                 max_rec_batch_size: int = 1,
                 max_det_batch_size: int = 1,
                 backend_name: str = 'trt',
                 force_fp16: bool = False,
                 triton_uri=None,
                 root_dir: str = '/models',
                 database_path: str = '',
                 **kwargs):

        if max_size is None:
            max_size = [640, 640]

        self.decode_required = True
        self.max_size = validate_max_size(max_size)
        self.max_rec_batch_size = max_batch_size
        self.max_det_batch_size = max_batch_size
        self.det_name = det_name
        self.rec_name = rec_name
        self.database_path = database_path

        if backend_name not in ('trt', 'triton') and max_rec_batch_size != 1:
            logging.warning('Batch processing supported only for TensorRT & Triton backend. Fallback to 1.')
            self.max_rec_batch_size = 1

        assert det_name is not None

        self.det_model = Detector(det_name=det_name, max_size=self.max_size,
                                  max_batch_size=self.max_det_batch_size, backend_name=backend_name,
                                  force_fp16=force_fp16, triton_uri=triton_uri, root_dir=root_dir)

        if rec_name is not None:
            self.rec_model = get_model(rec_name, backend_name=backend_name, force_fp16=force_fp16,
                                       max_batch_size=self.max_rec_batch_size, root_dir=root_dir,
                                       download_model=False, triton_uri=triton_uri)
            self.rec_model.prepare()
        else:
            self.rec_model = None
        
        #Warming
        try:
            image = np.random.randint(low=0, high=256, size=(1, 224, 224, 3), dtype=np.uint8)
            emb = self.get(image)
        except:
            print('Warmed up')


    def sort_boxes(self, boxes, probs, landmarks, shape, max_num=0):
        # Based on original InsightFace python package implementation
        if max_num > 0 and boxes.shape[0] > max_num:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            img_center = shape[0] // 2, shape[1] // 2
            offsets = np.vstack([
                (boxes[:, 0] + boxes[:, 2]) / 2 - img_center[1],
                (boxes[:, 1] + boxes[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            boxes = boxes[bindex, :]
            probs = probs[bindex]

            landmarks = landmarks[bindex, :]

        return boxes, probs, landmarks

    def process_faces(self,
                      faces: List[dict],
                      extract_embedding: bool = True,
                      extract_ga: bool = True,
                      return_face_data: bool = False,
                      detect_masks: bool = True,
                      mask_thresh: float = 0.89,
                      **kwargs):
        chunked_faces = to_chunks(faces, self.max_rec_batch_size)
        for chunk in chunked_faces:
            chunk = list(chunk)
            crops = [e['facedata'] for e in chunk]
            total = len(crops)
            embeddings = [None] * total
            ga = [[None, None]] * total

            if extract_embedding:
                t0 = time.perf_counter()
                embeddings = self.rec_model.get_embedding(crops)
                took = time.perf_counter() - t0
                logging.debug(
                    f'Embedding {total} faces took: {took * 1000:.3f} ms. ({(took / total) * 1000:.3f} ms. per face)')

            for i, crop in enumerate(crops):
                embedding_norm = None
                normed_embedding = None
                gender = None
                age = None
                mask = None
                mask_probs = None

                embedding = embeddings[i]
                if extract_embedding:
                    embedding_norm = norm(embedding)
                    normed_embedding = embedding / embedding_norm

                face = chunk[i]
                if return_face_data is False:
                    face['facedata'] = None

                # face['raw_vec'] = embedding
                face['norm'] = embedding_norm
                face['vec'] = normed_embedding
                face['gender'] = gender
                face['age'] = age
                face['mask'] = mask
                face['mask_probs'] = mask_probs

                yield face

    # Process single image
    
    def get(self, images,
                  extract_embedding: bool = True,
                  return_face_data: bool = True,
                  max_size: List[int] = None,
                  threshold: float = 0.6,
                  min_face_size: int = 0,
                  limit_faces: int = 0,
                  use_rotation: bool = False,
                  **kwargs):

        # If detector has input_shape attribute, use it instead of provided value
        try:
            max_size = self.det_model.retina.input_shape[2:][::-1]
        except:
            pass

        # Pre-assign max_size to resize function
        _partial_resize = partial(resize_image, max_size=max_size)
        # Pre-assign threshold to detect function
        _partial_detect = partial(self.det_model.detect, threshold=threshold)

        # Initialize resied images iterator
        res_images = map(_partial_resize, images)
        batches = to_chunks(res_images, self.max_det_batch_size)

        faces = []
        faces_per_img = {}

        for bid, batch in enumerate(batches):
            batch_imgs, scales = zip(*batch)
            det_predictions = zip(*_partial_detect(batch_imgs))

            for idx, pred in enumerate(det_predictions):
                orig_id = (bid * self.max_det_batch_size) + idx
                boxes, probs, landmarks = pred
                faces_per_img[orig_id] = len(boxes)

                if not isinstance(boxes, type(None)):
                    if limit_faces > 0:
                        boxes, probs, landmarks = self.sort_boxes(boxes, probs, landmarks,
                                                                  shape=batch_imgs[idx].shape,
                                                                  max_num=limit_faces)
                        faces_per_img[orig_id] = len(boxes)

                    # Translate points to original image size
                    boxes = reproject_points(boxes, scales[idx])

                    landmarks = reproject_points(landmarks, scales[idx])
                    # Crop faces from original image instead of resized to improve quality
                    if extract_embedding:
                        crops = face_align.norm_crop_batched(images[orig_id], landmarks)
                    else:
                        crops = [None] * len(boxes)

                    for i, _crop in enumerate(crops):
                        face = dict(
                            bbox=boxes[i], landmarks=landmarks[i], prob=probs[i],
                            num_det=i, scale=scales[idx], facedata=_crop
                        )
                        if min_face_size > 0:
                            w = boxes[i][2] - boxes[i][0]
                            if w >= min_face_size:
                                faces.append(face)
                        else:
                            faces.append(face)

        # Process detected faces
        if extract_embedding:
            faces = list(self.process_faces(faces,
                                            extract_embedding=extract_embedding,
                                            return_face_data=return_face_data,))
    
        faces_by_img = []
        offset = 0

        for key in faces_per_img:
            value = faces_per_img[key]
            faces_by_img.append(faces[offset:offset + value])
            offset += value

        return faces_by_img
    
    def update_db(self, face_embeddings):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as db_file:
                face_database = pickle.load(db_file)
            face_embeddings = face_database + face_embeddings
        with open(self.database_path, 'wb') as f:
            pickle.dump(face_embeddings, f)

        return True
    
    def register_person(self, data, step=20):
        face_embeddings = []
        video_path = data['video']
        person_info = data['person_info']
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    embedding = self.get([frame])[0]
                except:
                    continue
                if len(embedding) == 0:
                    continue
                emb = {}
                emb['embedding'] = embedding[0]['vec']
                emb['identity'] = person_info
                face_embeddings.append(emb)

        cap.release()
        status = self.update_db(face_embeddings=face_embeddings)

        return status
    
    def get_db_embeddings(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as db_file:
                face_database = pickle.load(db_file)
        else:
            face_database = {}
            return [], [], True

        embeddings_array = np.array([face['embedding'] for face in face_database])
        identities_array = np.array([face['identity'] for face in face_database])

        return embeddings_array, identities_array, False

    def face_recognition(self, batch, conf):
        embeddings_batch = self.get(batch)
        db_embeddings, db_identities, status_empty_db = self.get_db_embeddings()

        annots = []
        for x, embeddings in enumerate(embeddings_batch):
            frame = batch[x]

            if len(embeddings) == 0 or status_empty_db:  
                print(embeddings)
                annots.append(([[], [], ["Unknown"]]))
                continue
            print(embeddings)

            curr_embeddings = np.array([face['vec'] for face in embeddings])
            results = cosine_sim(curr_embeddings, db_embeddings)

            boxes, confs, prd_names = [], [], []
            for i in range(curr_embeddings):
                bbox = embeddings[i]['bbox'].astype(int)
                boxes.append(bbox)
                ind, conf_cosine = results[i]
                confs.append(conf_cosine)

                if conf_cosine > conf:
                    person_info = db_identities[int(ind)]
                else:
                    person_info = "Unknown"
                prd_names.append(person_info)

            annots.append([boxes, confs, person_info])

        return annots
