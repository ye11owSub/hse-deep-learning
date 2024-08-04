import numpy as np
from ultralytics import YOLOv10

from hse_deep_learning.deep_sort.deep_sort.detection import Detection
from hse_deep_learning.detectors.base import DetectionsProvider
from hse_deep_learning.utils.shapes import Rect

_LABEL_PERSON = 0


class YoloV10(DetectionsProvider):
    def __init__(self, model_name: str):
        self.model = YOLOv10.from_pretrained(f"jameslahm/{model_name}")
        self.model.to(self.device)

    def load_detections(self, image: np.ndarray, frame_id: int, min_height: int = 0) -> list[Detection]:
        results = self.model(image, verbose=False)
        detection_list = []

        for result in results:
            result = result

            for box in result.boxes:
                rect = box.xyxy[0].cpu().detach().numpy()
                x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]

                confidence = box.conf.cpu().detach().numpy()[0]
                label = box.cls.cpu().detach().numpy()[0]

                if label != _LABEL_PERSON:
                    continue

                width = x1 - x0
                height = y1 - y0

                if height < min_height:
                    continue

                rect = Rect(left=x0, top=y0, width=width, height=height)
                detection_list.append(Detection(rect, confidence))

        return detection_list
