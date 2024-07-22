from functools import cached_property
from typing import Optional

import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor

from hse_deep_learning.utils.shapes import Rect


class TorchReidFeaturesExtractor:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def crop_by_rectangle(
        self, image: np.ndarray, bbox: Rect, patch_shape: Optional[tuple[float, float]] = None
    ) -> np.ndarray:
        if patch_shape is not None:
            bbox = bbox.resize(target_width=patch_shape[0], target_height=patch_shape[1])

        image_width = image.shape[1]
        image_height = image.shape[0]

        image_box = Rect(left=0, top=0, width=image_width, height=image_height)

        bbox = image_box.clip(bbox)
        image_patch = image[int(bbox.top) : int(bbox.bottom), int(bbox.left) : int(bbox.right)]

        if patch_shape is not None:
            target_size = int(patch_shape[0]), int(patch_shape[1])
        else:
            target_size = int(bbox.width), int(bbox.height)

        image_patch = cv2.resize(image_patch, target_size)

        return image_patch

    @cached_property
    def extractor(self):
        return FeatureExtractor(model_name=self.model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def extract(self, image: np.ndarray, boxes: list[Rect]) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty(shape=(0, 1))

        croped_images = [self.crop_by_rectangle(image, box) for box in boxes]
        out = self.extractor(croped_images)
        return out.cpu().detach().numpy()
