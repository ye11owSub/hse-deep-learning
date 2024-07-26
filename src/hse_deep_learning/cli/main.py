from argparse import ArgumentParser
from pathlib import Path

import pkg_resources

from hse_deep_learning.app import GroundTruthApp
from hse_deep_learning.utils.dataset import load

DATA_PATH = pkg_resources.resource_filename("hse_deep_learning", "share/")


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-commands help")

    ground_truth_parser = subparsers.add_parser("ground-truth")
    ground_truth_parser.set_defaults(cmd="ground-truth")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cmd == "ground-truth":
        dataset = load(str(Path(DATA_PATH) / "dataset"))
        app = GroundTruthApp(dataset)
        app.run()


if __name__ == "__main__":
    main()
