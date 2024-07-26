import numpy as np
import torch

from hse_deep_learning.deep_sort.deep_sort.detection import Detection
from hse_deep_learning.detectors.base import DetectionsProvider
from hse_deep_learning.utils.shapes import Rect

_LABEL_PERSON = 0


class YoloV5(DetectionsProvider):
    MODELS_NAMES = {
        "yolov5l",
        "yolov5m",
        "yolov5n",
        "yolov5n6",
        "yolov5s",
    }

    def __init__(self, model_name: str):
        path = self.MODELS_FOLDER / "yolov5"
        if not path.exists() or model_name not in self.MODELS_NAMES:
            raise IOError

        self.model = torch.hub.load(str(path), "custom", source="local", trust_repo=True, path=str(path / model_name))
        self.model.to(self.device)

    def load_detections(self, image: np.ndarray, frame_id: int, min_height: int = 0) -> list[Detection]:
        # print(frame_id)
        results = self.model(image)
        detections = []

        for obj in results.pred[0]:
            x0, y0, x1, y1, confidence, label = obj.cpu().detach().numpy()

            if label != _LABEL_PERSON:
                continue

            width = x1 - x0
            height = y1 - y0

            if height < min_height:
                continue

            rect = Rect(left=x0, top=y0, width=width, height=height)
            detections.append(Detection(rect, confidence))

        return detections
