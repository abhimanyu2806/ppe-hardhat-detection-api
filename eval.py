"""
eval.py

Runs final evaluation of the trained RT-DETR PPE detection model on the
held-out test split, reporting overall and per-class precision, recall,
mAP50, and mAP50-95.

These are the official, final numbers reported in the memo (Section 3) --
computed on the test split, which was never used during training or for
any hyperparameter decisions.

Usage:
    python eval.py --weights model/best.pt --data path/to/data.yaml
"""

import argparse
import os
import sys

from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained RT-DETR PPE model on the test split.")

    parser.add_argument(
        "--weights",
        type=str,
        default="model/best.pt",
        help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset's data.yaml (must include a 'test' split). "
             "You must download the dataset first -- see README.md for the "
             "Roboflow download command -- then point this at YOUR local "
             "data.yaml path. This cannot be a placeholder path."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on. Defaults to 'test' -- the held-out "
             "split used for all final reported metrics, never touched during training."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"ERROR: Model weights not found at '{args.weights}'.")
        print("Download best.pt from the link in README.md and place it at the path above.")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"ERROR: data.yaml not found at '{args.data}'.")
        print("You need to download the dataset first -- see the 'Reproducing training "
              "and evaluation' section in README.md for the exact Roboflow download command.")
        print("Then pass the path to YOUR downloaded data.yaml via --data.")
        sys.exit(1)

    print(f"Loading model from {args.weights}")
    model = RTDETR(args.weights)

    print(f"Running evaluation on the '{args.split}' split of {args.data}")
    metrics = model.val(data=args.data, split=args.split)

    class_names = model.names

    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    print(f"Precision (mean): {metrics.box.mp:.4f}")
    print(f"Recall (mean):    {metrics.box.mr:.4f}")
    print(f"mAP50:            {metrics.box.map50:.4f}")
    print(f"mAP50-95:         {metrics.box.map:.4f}")

    print("\n" + "=" * 50)
    print("PER-CLASS METRICS")
    print("=" * 50)

    per_class_map50 = metrics.box.maps  # per-class mAP50-95, indexed by class id
    precisions = metrics.box.p
    recalls = metrics.box.r

    for class_id, class_name in class_names.items():
        print(
            f"{class_name:15s} | "
            f"Precision: {precisions[class_id]:.4f} | "
            f"Recall: {recalls[class_id]:.4f} | "
            f"mAP50-95: {per_class_map50[class_id]:.4f}"
        )

    print("\nFull results (plots, confusion matrix, etc.) saved under "
          "the runs/ directory created by Ultralytics.")


if __name__ == "__main__":
    main()