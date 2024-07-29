import abc
from pathlib import Path

import numpy as np
import torch

from hse_deep_learning.deep_sort.deep_sort.detection import Detection


class DetectionsProvider(abc.ABC):
    MODELS_FOLDER = Path("share") / "models"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def load_detections(self, image: np.ndarray, frame_id: int, min_height: int = 0) -> list[Detection]: ...
