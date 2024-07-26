import cv2

from hse_deep_learning.custom_deep_sort import CustomDeepSort
from hse_deep_learning.utils.dataset import DatasetProcessor
from hse_deep_learning.utils.painter import Painter


class App:
    def __init__(
        self,
        dataset_descriptor: DatasetProcessor,
        deep_sort: CustomDeepSort,
    ):
        frame_shape = dataset_descriptor.image_size[::-1]
        aspect_ratio = float(frame_shape[1]) / frame_shape[0]

        self.window_shape = 1024, int(aspect_ratio * 1024)
        self.title = dataset_descriptor.name
        self.deep_sort = deep_sort
        self.images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in dataset_descriptor.images_files]
        update_rate_ms = dataset_descriptor.update_rate
        if update_rate_ms is None:
            update_rate_ms = 5

        self.fps_records: list[float] = list()

    def destroy_app(self):
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)

    def run(self):
        for frame_id, image in enumerate(self.images):
            if image is None:
                self.destroy_app()
                return

            tracks, detections = self.deep_sort.update(frame_id, image)

            painter = Painter(image)
            painter.draw_detections(detections)
            painter.draw_trackers(tracks)
            cv2.imshow(self.title, cv2.resize(image, self.window_shape))
