from typing import Optional

import numpy as np

from hse_deep_learning.deep_sort.deep_sort import nn_matching
from hse_deep_learning.deep_sort.deep_sort.track import Track
from hse_deep_learning.deep_sort.deep_sort.tracker import Tracker
from hse_deep_learning.detectors.base import Detection, DetectionsProvider
from hse_deep_learning.features_extractors.tourch_reid import TorchReidFeaturesExtractor
from hse_deep_learning.utils.shapes import Rect


class CustomDeepSort:
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_MAX_OVERLAP = 1.0
    TRACKING_MAX_IOU_DISTANCE = 0.7
    TRACKING_MAX_AGE = 30
    TRACKING_N_INIT = 3

    def __init__(self, detections_provider: DetectionsProvider, features_extractor: TorchReidFeaturesExtractor):
        self.detections_provider = detections_provider
        self.features_extractor = features_extractor
        self.tracker = Tracker(
            metric=nn_matching.NearestNeighborDistanceMetric("cosine", 0.2),
            max_iou_distance=self.TRACKING_MAX_IOU_DISTANCE,
            max_age=self.TRACKING_MAX_AGE,
            n_init=self.TRACKING_N_INIT,
        )

    def non_max_suppression(self, boxes: np.ndarray, max_bbox_overlap: float, scores: Optional[np.ndarray] = None):
        if len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float32)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + boxes[:, 0]
        y2 = boxes[:, 3] + boxes[:, 1]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores) if scores is not None else np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))

        return pick

    def update(self, frame_id: int, image: np.ndarray) -> tuple[list[Track], list[Detection]]:
        detections = self.detections_provider.load_detections(image, frame_id)
        first_stage_results: list[Detection] = detections

        detections = [detection for detection in detections if detection.confidence >= self.DETECTION_MIN_CONFIDENCE]
        bounding_boxes = np.array(
            [
                [detection.origin.left, detection.origin.top, detection.origin.width, detection.origin.height]
                for detection in detections
            ]
        )
        confidence_scores = np.array([d.confidence for d in detections])

        indices = self.non_max_suppression(bounding_boxes, self.DETECTION_NMS_MAX_OVERLAP, confidence_scores)

        filtered_detections_bboxes: list[Rect] = [detections[i].origin for i in indices]

        extracted_features = self.features_extractor.extract(image, filtered_detections_bboxes)

        self.tracker.predict()
        self.tracker.update(filtered_detections_bboxes, extracted_features)

        return self.tracker.tracks, first_stage_results
