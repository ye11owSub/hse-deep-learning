import numpy as np

from hse_deep_learning.deep_sort.deep_sort import nn_matching, preprocessing
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

    @property
    def tracker(self) -> Tracker:
        return Tracker(
            metric=nn_matching.NearestNeighborDistanceMetric("cosine", 0.2),
            max_iou_distance=self.TRACKING_MAX_IOU_DISTANCE,
            max_age=self.TRACKING_MAX_AGE,
            n_init=self.TRACKING_N_INIT,
        )

    def update(self, frame_id: int, image: np.ndarray) -> tuple[list[Track], list[Detection]]:
        detections = self.detections_provider.load_detections(image, frame_id)
        first_stage_results: list[Detection] = detections

        detections = [detection for detection in detections if detection.confidence >= self.DETECTION_MIN_CONFIDENCE]
        bounding_boxes = np.array([list(detection.origin) for detection in detections])
        confidence_scores = np.array([d.confidence for d in detections])

        indices = preprocessing.non_max_suppression(bounding_boxes, self.DETECTION_NMS_MAX_OVERLAP, confidence_scores)

        filtered_detections_bboxes: list[Rect] = [detections[i].origin for i in indices]

        extracted_features = self.features_extractor.extract(image, filtered_detections_bboxes)

        self.tracker.predict()
        self.tracker.update(filtered_detections_bboxes, extracted_features)

        return self.tracker.tracks, first_stage_results
