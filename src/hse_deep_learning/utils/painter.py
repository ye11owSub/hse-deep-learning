import colorsys

import cv2
import numpy as np
from pydantic import BaseModel

from hse_deep_learning.deep_sort.deep_sort.detection import Detection
from hse_deep_learning.deep_sort.deep_sort.track import Track
from hse_deep_learning.utils.shapes import Rect


class Color(BaseModel):
    red: int
    green: int
    blue: int


class Brush(BaseModel):
    thickness: int
    text_size: int
    color: Color


class Painter:
    def __init__(self, image: np.ndarray):
        self.image = image.copy()

    @property
    def output_image(self) -> np.ndarray:
        return self.image

    def measure_text(self, text: str, brush: Brush) -> tuple[int, int]:
        width = 0
        height = 0

        for line in text.split("\n"):
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, brush.text_size, brush.thickness)
            width = max(width, size[0][0])
            height += size[0][1]

        return width, height

    def add_rectangle(self, x: int, y: int, width: int, height: int, brush: Brush):
        color = brush.color
        cv2.rectangle(
            self.image,
            (x, y),
            (x + width, y + height),
            (
                color.red,
                color.green,
                color.blue,
            ),
            brush.thickness,
        )

    def add_text(self, x: int, y: int, text: str, brush: Brush):
        color = brush.color
        for line in reversed(text.split("\n")):
            _, line_height = self.measure_text(line, brush)

            cv2.putText(
                self.image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_PLAIN,
                brush.text_size,
                (
                    color.red,
                    color.green,
                    color.blue,
                ),
                brush.thickness,
            )
            y -= line_height

    def create_unique_color(self, tag: int, hue_step: float = 0.41) -> Color:
        hue, value = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0

        red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, value)
        return Color(red=int(red * 255), green=int(green * 255), blue=int(blue * 255))

    def draw_label_with_bounding_box(self, label: str, x: int, y: int, padding: int, brush: Brush):
        label_width, label_height = self.measure_text(label, brush)
        label_x, label_y = x + padding, y + padding + label_height

        width = label_width + 2 * padding
        height = label_height + 2 * padding

        self.add_rectangle(x, y, width, height, brush)
        self.add_text(label_x, label_y, label, brush)

    def draw_track(self, track_id: int, box: Rect):
        track_color: Color = self.create_unique_color(track_id)

        self.add_rectangle(
            int(box.left),
            int(box.top),
            int(box.width),
            int(box.height),
            brush=Brush(thickness=3, text_size=1, color=track_color),
        )

        self.draw_label_with_bounding_box(
            label=str(track_id),
            x=int(box.left),
            y=int(box.top),
            padding=5,
            brush=Brush(thickness=2, text_size=1, color=track_color),
        )

    def draw_detections(self, detections: list[Detection]):
        for detection in detections:
            detection_bbox = detection.origin
            self.add_rectangle(
                int(detection_bbox.left),
                int(detection_bbox.top),
                int(detection_bbox.width),
                int(detection_bbox.height),
                brush=Brush(thickness=3, text_size=1, color=Color(red=0, green=0, blue=255)),
            )

    def draw_trackers(self, tracks: list[Track]):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            self.draw_track(track.track_id, track.bounding_box)

    def draw_info(self, info: str):
        self.draw_label_with_bounding_box(
            label=info,
            x=25,
            y=25,
            padding=5,
            brush=Brush(thickness=2, text_size=2, color=Color(red=0, green=0, blue=0)),
        )
