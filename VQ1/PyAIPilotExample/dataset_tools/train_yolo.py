"""
Fine-tune a YOLOv8-pose model for gate detection on the auto-labeled dataset
(see build_dataset.py).

ultralytics re-initializes the pose head for this dataset's kpt_shape: [4, 3]
(4 gate-corner keypoints) vs. COCO's 17-keypoint head, while transferring the
pretrained backbone -- standard fine-tuning workflow.

Usage:
    python train_yolo.py [--epochs 100] [--imgsz 640] [--batch 16]

If yolov8n-pose accuracy is insufficient, retry with --model yolov8s-pose.pt
(more capacity, slower).
"""

import argparse
import os

from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PyAIPilotExample/
DATA_YAML = os.path.join(REPO_ROOT, 'dataset', 'data.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yolov8n-pose.pt',
                         help='Base checkpoint (default yolov8n-pose.pt)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default=None,
                         help='cuda device, e.g. "0"; default auto-selects GPU if available')
    parser.add_argument('--project', default=os.path.join(REPO_ROOT, 'runs', 'pose'),
                         help='Output directory for training runs')
    parser.add_argument('--name', default='gate_yolo')
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f'{DATA_YAML} not found -- run build_dataset.py first')

    model = YOLO(args.model)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
