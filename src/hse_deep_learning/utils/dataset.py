import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from hse_deep_learning.utils.shapes import Rect


class GroundTruth:
    def __init__(self, ground_truth_file: str):
        self.raw_data = np.loadtxt(ground_truth_file, delimiter=",")
        self.frames_lookup: dict[int, dict[int, Rect]] = dict()
        self.tracks_lookup: dict[int, list[tuple[int, Rect]]] = dict()

        for entry in self.raw_data:
            frame_id, track_id, bbox, confidence = int(entry[0]), int(entry[1]), entry[2:6], entry[6]

            should_consider = confidence == 1

            if not should_consider:
                continue

            if frame_id not in self.frames_lookup:
                self.frames_lookup[frame_id] = dict()

            if track_id not in self.tracks_lookup:
                self.tracks_lookup[track_id] = list()

            bbox_rect = Rect(left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3])
            self.frames_lookup[frame_id][track_id] = bbox_rect
            self.tracks_lookup[track_id].append((frame_id, bbox_rect))

    def __getitem__(self, frame_id: int) -> dict[int, Rect]:
        return self.frames_lookup.get(frame_id, dict())

    def get_track(self, track_id: int) -> list[tuple[int, Rect]]:
        return self.tracks_lookup[track_id]


class DatasetProcessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    images_files: List[str]
    ground_truth: Optional[GroundTruth]
    detections: np.ndarray
    image_size: np.ndarray
    update_rate: float


def load(sequence_directory_path: str) -> DatasetProcessor:
    images_directory = Path(sequence_directory_path) / "img1"
    images_files = sorted([os.path.join(images_directory, file) for file in os.listdir(images_directory)])
    ground_truth_file = Path(sequence_directory_path) / "gt" / "gt.txt"

    ground_truth = GroundTruth(str(ground_truth_file))

    detection_file = os.path.join(sequence_directory_path, "det", "det.txt")
    detections = np.loadtxt(detection_file, delimiter=",")

    image = cv2.imread(images_files[0], cv2.IMREAD_GRAYSCALE)
    image_size = np.array(image.shape)

    info_file = os.path.join(sequence_directory_path, "seqinfo.ini")
    with open(info_file, "r") as file:
        line_splits = [line.split("=") for line in file.read().splitlines()[1:]]
        info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

    update_ms = 1000 / int(info_dict["frameRate"])

    return DatasetProcessor(
        name=os.path.basename(sequence_directory_path),
        images_files=images_files,
        ground_truth=ground_truth,
        detections=detections,
        image_size=image_size,
        update_rate=update_ms,
    )
