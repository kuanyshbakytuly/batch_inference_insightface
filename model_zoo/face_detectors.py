from .detectors.scrfd import SCRFD


def scrfd(model_path, backend, device_id, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, device_id=device_id, **kwargs)
    model = SCRFD(inference_backend=inference_backend)
    return model

def scrfd_v2(model_path, backend, device_id, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, device_id=device_id, **kwargs)
    model = SCRFD(inference_backend=inference_backend, ver=2)
    return model
