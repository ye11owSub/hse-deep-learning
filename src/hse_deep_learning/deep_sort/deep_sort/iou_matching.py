# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np

from hse_deep_learning.deep_sort.deep_sort.track import Track
from hse_deep_learning.utils.shapes import Rect

from . import linear_assignment


def get_rect_untersection(that: Rect, other: Rect):
    that_area = that.area
    other_area = that.area

    intersection_top = max(that.top, other.top)
    intersection_left = max(that.left, other.left)
    intersection_bottom = min(that.bottom, other.bottom)
    intersection_right = min(that.right, other.right)

    intersection_width = intersection_right - intersection_left
    intersection_height = intersection_bottom - intersection_top

    intersection_area = intersection_width * intersection_height

    union_area = that_area + other_area - intersection_area
    return intersection_area / union_area


def iou(bbox: Rect, candidates: list[Rect]) -> np.ndarray:
    return np.array([get_rect_untersection(bbox, candidate) for candidate in candidates])


def iou_cost(
    tracks: list[Track],
    detections_bboxes: list[Rect],
    features: np.ndarray,
    track_indices: list[int],
    detection_indices: list[int],
):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].bounding_box
        candidates = [detections_bboxes[i] for i in detection_indices]
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix
