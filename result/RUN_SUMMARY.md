# YOLO-Box four-dataset experiment results

Generated from fresh runs on 2026-09-15. All runs used YOLOv5n pretrained weights,
30 epochs, batch size 2, image size 640, CPU, workers 0, seed 2569, and the default
`hyp.scratch-low.yaml`. The four datasets are deterministic nested subsets.

## Common-test comparison

Every `best.pt` was evaluated on the same `f09_box_350` test split: 50 images and
351 annotated objects.

| Dataset images | Iterations | Best epoch | Best val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 330 | 28 | 0.09685 | 0.141 | 0.318 | 0.212 | 0.0706 |
| 60 | 630 | 18 | 0.33475 | 0.365 | 0.708 | 0.546 | 0.2490 |
| 150 | 1620 | 28 | 0.70766 | 0.886 | 0.987 | 0.984 | 0.6520 |
| 350 | 3750 | 29 | 0.81224 | 0.990 | 0.990 | 0.989 | 0.7250 |

The size-350 run is the winner by common-test mAP50-95
(0.7250). Performance improves strongly as the training
subset grows, with the largest gain occurring between 60 and 150 images.

## TensorBoard event verification

- Size 30: 330 points/tag, counts=True, steps=True, finite=True, max total-sum error=2.235e-08.
- Size 60: 630 points/tag, counts=True, steps=True, finite=True, max total-sum error=1.863e-08.
- Size 150: 1620 points/tag, counts=True, steps=True, finite=True, max total-sum error=2.235e-08.
- Size 350: 3750 points/tag, counts=True, steps=True, finite=True, max total-sum error=1.863e-08.

For every run, `train/total_loss` equals box + objectness + classification loss up
to normal floating-point rounding. Each run has exactly 30 CSV rows and both
`weights/best.pt` and `weights/last.pt`.

## Evidence images

- `screenshots/00_tensorboard_ui.png`: actual TensorBoard UI showing the completed runs.
- `screenshots/01_tensorboard_iteration_losses.png`: component and total losses at every iteration.
- `screenshots/02_tensorboard_validation_metrics.png`: precision, recall, mAP50, and mAP50-95 by epoch.
- `screenshots/03_tensorboard_learning_rate.png`: scheduled learning rate by iteration.
- `screenshots/04_common_test_comparison.png`: fair comparison on the shared test split.
- `screenshots/05_inference_first.jpg` through `07_inference_last.jpg`: predictions from the winning model.
- `screenshots/per_run/size*_tensorboard_dashboard.png`: separate loss, LR, and metric dashboard for each run.
- `screenshots/per_run/size*_yolov5_results.png`: native YOLOv5 epoch plots for each run.

The first and last samples detect two lanes, three track-lines, and two sideway
regions. The middle sample detects the same scene structure, although one right-side
region is labeled track-line at lower confidence. The boxes remain spatially
consistent across the sequence; their large overlap reflects the dataset's broad,
overlapping region annotations.

Raw common-test console output is saved as `common_test_size*.txt`. Training runs,
event files, plots, and checkpoints are under `yolov5/runs/f09_box_compare/`.
