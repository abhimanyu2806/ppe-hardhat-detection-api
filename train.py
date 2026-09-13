"""
train.py

Fine-tunes RT-DETR on the Hard Hats PPE dataset (Hardhat / NO-Hardhat).

This script matches the exact training run used to produce the submitted
model weights (best.pt). It was originally run in Google Colab on a free-tier
NVIDIA T4 GPU, split across two sessions due to compute-time limits (see
README.md and the memo for the full reproducibility notes, including a
training divergence encountered near epoch 22).

Usage:
    # Fresh training run
    python train.py --data path/to/data.yaml

    # Resume an interrupted run (e.g. after hitting a Colab session limit)
    python train.py --resume path/to/weights/last.pt

Dataset:
    Hard Hats Dataset, Roboflow Universe
    https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5
    (CC BY 4.0, version 1, YOLOv8 export format)

    Download it yourself with:
        pip install roboflow
        from roboflow import Roboflow
        rf = Roboflow(api_key="YOUR_KEY")
        project = rf.workspace("roboflow-universe-projects").project("hard-hats-fhbh5")
        version = project.version(1)
        dataset = version.download("yolov8")
"""

import argparse
import os
import sys

from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on the PPE Hard Hats dataset.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset's data.yaml (train/val/test paths + class names). "
             "You must download the dataset first -- see README.md for the "
             "Roboflow download command -- then point this at YOUR local data.yaml path."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a last.pt checkpoint to resume an interrupted training run from."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Kept small (8) to fit a free-tier T4 GPU's memory."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience (epochs without validation improvement)."
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs, in addition to best.pt/last.pt. "
             "Important for free-tier Colab, where sessions can disconnect mid-run."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory where training runs (weights, logs, plots) are saved."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ppe_rtdetr_run",
        help="Name of this specific training run's output folder."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="rtdetr-l.pt",
        help="Pretrained RT-DETR checkpoint to fine-tune from, if not resuming."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Resume checkpoint not found at '{args.resume}'.")
            sys.exit(1)
        # Resuming reads epoch count, data path, and all other settings
        # from the original run's args.yaml automatically.
        print(f"Resuming training from checkpoint: {args.resume}")
        model = RTDETR(args.resume)
        model.train(resume=True, save_period=args.save_period)

    else:
        if not os.path.exists(args.data):
            print(f"ERROR: data.yaml not found at '{args.data}'.")
            print("You need to download the dataset first -- see the dataset download "
                  "command in README.md / at the top of this file. Then pass the path "
                  "to YOUR downloaded data.yaml via --data.")
            sys.exit(1)

        print(f"Starting fresh training run using {args.checkpoint}")
        model = RTDETR(args.checkpoint)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            save_period=args.save_period,
            project=args.project,
            name=args.name,
        )

    print("Training complete. Best checkpoint saved under "
          f"{args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()