import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel


class DatasetProcessor(BaseModel):
    name: str
    images_files: List[str]
    ground_truth: Optional[MotGroundTruth]
    detections: np.ndarray
    image_size: Optional[np.ndarray]
    update_rate: Optional[float]


def load(sequence_directory: str) -> DatasetProcessor:
    images_directory = Path(sequence_directory) / "img1"
    images_files = sorted([os.path.join(images_directory, file) for file in os.listdir(images_directory)])
    ground_truth_file = Path(sequence_directory) / "gt" / "gt.txt"

    ground_truth = MotGroundTruth.load(ground_truth_file)

    detection_file = os.path.join(sequence_directory, "det", "det.txt")
    detections = np.loadtxt(detection_file, delimiter=",")

    if len(images_files) > 0:
        image = cv2.imread(next(iter(images_files)), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    info_file = os.path.join(sequence_directory, "seqinfo.ini")
    if os.path.exists(info_file):
        with open(info_file, "r") as file:
            line_splits = [line.split("=") for line in file.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    return DatasetProcessor(
        name=os.path.basename(sequence_directory),
        images_files=images_files,
        ground_truth=ground_truth,
        detections=detections,
        image_size=image_size,
        update_rate=update_ms,
    )
