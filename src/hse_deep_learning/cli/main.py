import itertools
from argparse import ArgumentParser
from pathlib import Path

from hse_deep_learning.app import App, GroundTruthApp
from hse_deep_learning.custom_deep_sort import CustomDeepSort
from hse_deep_learning.detectors.yolov5 import YoloV5
from hse_deep_learning.features_extractors.tourch_reid import TorchReidFeaturesExtractor
from hse_deep_learning.metrics import Metrics
from hse_deep_learning.utils.dataset import load

FEATURES_EXTRACTORS = {
    "shufflenet",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "mlfn",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "osnet_ain_x0_75",
}


DETECTORS = {"yolov5n", "yolov5n6", "yolov5s", "yolov5m", "yolov5l"}


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-commands help")

    ground_truth_parser = subparsers.add_parser("ground-truth")
    ground_truth_parser.set_defaults(cmd="ground-truth")

    deep_sort_parser = subparsers.add_parser("run", help="Runs Extended Deep SORT algorithm.")
    deep_sort_parser.set_defaults(cmd="run")

    deep_sort_parser.add_argument(
        "-d",
        "--detections_provider",
        help="Detections provider for finding human.",
        default=None,
        choices=DETECTORS,
        required=False,
    )

    deep_sort_parser.add_argument(
        "-fe",
        "--features_extractor",
        help=f"Features extractor for ReID.",
        default=None,
        choices=FEATURES_EXTRACTORS,
        required=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    datasets = Path("share") / "datasets"

    if args.cmd == "ground-truth":
        for dataset_path in datasets.glob("*"):
            dataset = load(str(dataset_path))
            app = GroundTruthApp(dataset)
            app.run()

    elif args.cmd == "run":
        for detector, extractor in itertools.product(DETECTORS, FEATURES_EXTRACTORS):
            print(f"\x1b[6;30;42m{detector}, {extractor}\x1b[0m")
            for dataset_path in datasets.glob("*"):
                dataset = load(str(dataset_path))
                deep_sort = CustomDeepSort(
                    detections_provider=YoloV5(detector),
                    features_extractor=TorchReidFeaturesExtractor(extractor),
                )
                metrics = Metrics(ground_truth=dataset.ground_truth)
                app = App(dataset_descriptor=dataset, deep_sort=deep_sort, metrics=metrics)
                app.run()


if __name__ == "__main__":
    main()
