from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from hse_deep_learning.deep_sort.deep_sort.iou_matching import get_rect_untersection
from hse_deep_learning.utils.dataset import GroundTruth
from hse_deep_learning.utils.shapes import Rect


class Metrics:
    def __init__(self, ground_truth: GroundTruth, iou_threshold: float = 0.5):
        self.ground_truth = ground_truth
        self.detections: dict[int, dict[int, Rect]] = dict()
        self.tracks: defaultdict[int, list[tuple[int, Rect]]] = defaultdict(list)
        self.iou_threshold = iou_threshold

    def evaluate(self) -> dict[str, float]:
        result_metrics: dict[str, float] = dict()
        overall_tp, overall_fp, overall_fn = 0, 0, 0

        for frame_id in self.detections.keys():
            tp, fp, fn = self.evaluate_frame(frame_id)

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        precision = 0.0
        precision_denominator = overall_tp + overall_fp

        recall = 0.0
        recall_denominator = overall_tp + overall_fn

        if overall_tp == 0 and precision_denominator == 0:
            precision = 1.0
        else:
            precision = overall_tp / precision_denominator

        if overall_tp == 0 and recall_denominator == 0:
            recall = 1.0
        else:
            recall = overall_tp / recall_denominator

        f1_score = 0.0
        f1_denominator = precision + recall

        if 2 * precision * recall == 0 and f1_denominator == 0:
            f1_score = 1.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        result_metrics["precision"] = precision
        result_metrics["recall"] = recall
        result_metrics["f1"] = f1_score

        return result_metrics

    def update(self, frame_id: int, detections: dict[int, Rect]):
        self.detections[frame_id] = detections

        for track_id, box in detections.items():
            self.tracks[track_id].append((frame_id, box))

    def evaluate_frame(self, frame_id: int) -> tuple[int, int, int]:
        raw_detections = self.detections[frame_id]
        raw_ground_truth = self.ground_truth[frame_id]

        if len(raw_detections) == 0 and len(raw_ground_truth) == 0:
            return 0, 0, 0
        elif len(raw_detections) == 0:
            return 0, 0, len(raw_ground_truth)
        elif len(raw_ground_truth) == 0:
            return 0, len(raw_detections), 0

        detection_ids, detection_boxes = zip(*raw_detections.items())
        ground_truth_ids, ground_truth_boxes = zip(*raw_ground_truth.items())

        scores = np.zeros((len(detection_boxes), len(ground_truth_boxes)), dtype=np.float32)

        for i in range(len(detection_ids)):
            detection_box = detection_boxes[i]
            for j in range(len(ground_truth_ids)):
                ground_truth_box = ground_truth_boxes[j]

                scores[i, j] = get_rect_untersection(detection_box, ground_truth_box)

        row_indexes, col_indexes = linear_sum_assignment(scores, maximize=True)

        tp, fp, fn = 0, 0, 0

        for row, column in zip(row_indexes, col_indexes):
            if scores[row, column] >= self.iou_threshold:
                tp += 1

        fp = len(detection_ids) - tp
        fn = len(ground_truth_ids) - tp

        return tp, fp, fn
