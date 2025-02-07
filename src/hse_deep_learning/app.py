import cv2

from hse_deep_learning.custom_deep_sort import CustomDeepSort
from hse_deep_learning.metrics import Metrics
from hse_deep_learning.utils.dataset import DatasetProcessor
from hse_deep_learning.utils.painter import Painter


class App:
    def __init__(
        self,
        dataset_descriptor: DatasetProcessor,
        deep_sort: CustomDeepSort,
        metrics: Metrics,
    ):
        frame_shape = dataset_descriptor.image_size[::-1]
        aspect_ratio = float(frame_shape[1]) / frame_shape[0]

        self.window_shape = 1024, int(aspect_ratio * 1024)
        self.title = dataset_descriptor.name
        self.deep_sort = deep_sort
        self.metrics = metrics
        self.dataset_descriptor = dataset_descriptor
        self.images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in dataset_descriptor.images_files]
        self.update_rate_ms = dataset_descriptor.update_rate

        self.fps_records: list[float] = list()

    def destroy_app(self):
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)

    def run(self) -> None:
        for frame_id, image in enumerate(self.images):
            if image is None:
                self.destroy_app()
                return

            painter = Painter(image)
            tracks, detections = self.deep_sort.update(frame_id, image)
            self.metrics.update(
                frame_id,
                {
                    track.track_id: track.bounding_box
                    for track in tracks
                    if track.is_confirmed() and track.time_since_update <= 1
                },
            )

            painter.draw_detections(detections)
            painter.draw_trackers(tracks)
            cv2.waitKey(int(self.update_rate_ms))
            cv2.imshow(self.title, cv2.resize(painter.output_image, self.window_shape))
        print({self.dataset_descriptor.name: self.metrics.evaluate()})

class GroundTruthApp:
    def __init__(
        self,
        dataset_descriptor: DatasetProcessor,
    ):
        frame_shape = dataset_descriptor.image_size[::-1]
        aspect_ratio = float(frame_shape[1]) / frame_shape[0]

        self.window_shape = 1024, int(aspect_ratio * 1024)
        self.title = dataset_descriptor.name
        self.dataset_descriptor = dataset_descriptor
        self.images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in dataset_descriptor.images_files]
        self.update_rate_ms = dataset_descriptor.update_rate

        self.fps_records: list[float] = list()

    def destroy_app(self):
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)

    def run(self):
        for frame_id, image in enumerate(self.images):
            if image is None:
                self.destroy_app()
                return
            painter = Painter(image)
            ground_truth = self.dataset_descriptor.ground_truth
            if ground_truth is None:
                raise
            tracks = ground_truth[frame_id]
            for track_id, bbox in tracks.items():
                painter.draw_track(track_id, bbox)
            cv2.waitKey(int(self.update_rate_ms))
            cv2.imshow(self.title, cv2.resize(painter.output_image, self.window_shape))
