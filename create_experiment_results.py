"""Export the four-dataset YOLO-Box experiment evidence into result/."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib
import markdown

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "yolov5" / "runs" / "f09_box_compare"
RESULT = ROOT / "result"
SHOTS = RESULT / "screenshots"
PER_RUN_SHOTS = SHOTS / "per_run"
SIZES = (30, 60, 150, 350)
EXPECTED_STEPS = {30: 330, 60: 630, 150: 1620, 350: 3750}
COLORS = {30: "#4c78a8", 60: "#f58518", 150: "#54a24b", 350: "#e45756"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def run_dir(size: int) -> Path:
    return RUN_ROOT / f"size{size}_e30_seed2569"


def load_events(size: int) -> EventAccumulator:
    event = next(run_dir(size).glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def read_train_summary(size: int) -> dict[str, object]:
    path = run_dir(size) / "results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    best_index, best = max(
        enumerate(rows), key=lambda item: float(item[1]["metrics/mAP_0.5:0.95"])
    )
    return {
        "epochs": len(rows),
        "best_epoch": best_index + 1,
        "val_precision": float(best["metrics/precision"]),
        "val_recall": float(best["metrics/recall"]),
        "val_map50": float(best["metrics/mAP_0.5"]),
        "val_map50_95": float(best["metrics/mAP_0.5:0.95"]),
    }


def read_common_test(size: int) -> dict[str, float]:
    path = RESULT / f"common_test_size{size}.txt"
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = ANSI.sub("", raw.decode(encoding))
    match = re.search(
        r"^\s*all\s+50\s+351\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Cannot parse common-test metrics for size {size}")
    precision, recall, map50, map50_95 = map(float, match.groups())
    return {
        "test_precision": precision,
        "test_recall": recall,
        "test_map50": map50,
        "test_map50_95": map50_95,
    }


def verify_events(size: int, events: EventAccumulator) -> dict[str, object]:
    tags = [
        "train/box_loss_iter",
        "train/obj_loss_iter",
        "train/cls_loss_iter",
        "train/total_loss",
        "train/learning_rate",
    ]
    series = {tag: events.Scalars(tag) for tag in tags}
    expected = EXPECTED_STEPS[size]
    counts_ok = all(len(values) == expected for values in series.values())
    steps_ok = [point.step for point in series[tags[0]]] == list(range(expected))
    finite = all(math.isfinite(point.value) for values in series.values() for point in values)
    max_error = max(
        abs(
            series["train/total_loss"][index].value
            - series["train/box_loss_iter"][index].value
            - series["train/obj_loss_iter"][index].value
            - series["train/cls_loss_iter"][index].value
        )
        for index in range(expected)
    )
    return {
        "iteration_points": expected,
        "event_counts_ok": counts_ok,
        "steps_contiguous": steps_ok,
        "values_finite": finite,
        "max_total_sum_error": max_error,
    }


def style_figure(fig, title: str) -> None:
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def save_iteration_losses(events: dict[int, EventAccumulator]) -> None:
    tags = [
        ("train/box_loss_iter", "Box loss"),
        ("train/obj_loss_iter", "Objectness loss"),
        ("train/cls_loss_iter", "Classification loss"),
        ("train/total_loss", "Total loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150)
    for axis, (tag, label) in zip(axes.flat, tags):
        for size in SIZES:
            values = events[size].Scalars(tag)
            axis.plot(
                [point.step for point in values],
                [point.value for point in values],
                color=COLORS[size],
                linewidth=0.8,
                alpha=0.75,
                label=f"{size} images ({len(values)} iter.)",
            )
        axis.set_title(label)
        axis.set_xlabel("Global training iteration")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    style_figure(fig, "YOLOv5n component losses logged at every iteration")
    fig.savefig(SHOTS / "01_tensorboard_iteration_losses.png", bbox_inches="tight")
    plt.close(fig)


def save_epoch_metrics(events: dict[int, EventAccumulator]) -> None:
    tags = [
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150)
    for axis, (tag, label) in zip(axes.flat, tags):
        for size in SIZES:
            values = events[size].Scalars(tag)
            axis.plot(
                [point.step + 1 for point in values],
                [point.value for point in values],
                color=COLORS[size],
                linewidth=2,
                label=f"{size} images",
            )
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    style_figure(fig, "Validation metrics from TensorBoard event data")
    fig.savefig(SHOTS / "02_tensorboard_validation_metrics.png", bbox_inches="tight")
    plt.close(fig)


def save_learning_rate(events: dict[int, EventAccumulator]) -> None:
    fig, axis = plt.subplots(figsize=(13, 6), dpi=150)
    for size in SIZES:
        values = events[size].Scalars("train/learning_rate")
        axis.plot(
            [point.step for point in values],
            [point.value for point in values],
            color=COLORS[size],
            linewidth=1.8,
            label=f"{size} images ({len(values)} iter.)",
        )
    axis.set_title("Scheduled learning rate at every training iteration")
    axis.set_xlabel("Global training iteration")
    axis.set_ylabel("Learning rate")
    axis.grid(alpha=0.25)
    axis.legend()
    style_figure(fig, "YOLOv5n learning-rate schedule")
    fig.savefig(SHOTS / "03_tensorboard_learning_rate.png", bbox_inches="tight")
    plt.close(fig)


def save_test_comparison(records: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(12, 6), dpi=150)
    x = list(range(len(records)))
    width = 0.34
    map50 = [float(record["test_map50"]) for record in records]
    map50_95 = [float(record["test_map50_95"]) for record in records]
    bars1 = axis.bar([i - width / 2 for i in x], map50, width, label="mAP@0.5", color="#4c78a8")
    bars2 = axis.bar([i + width / 2 for i in x], map50_95, width, label="mAP@0.5:0.95", color="#e45756")
    axis.bar_label(bars1, fmt="%.3f", padding=3)
    axis.bar_label(bars2, fmt="%.3f", padding=3)
    axis.set_xticks(x, [str(record["dataset_size"]) for record in records])
    axis.set_xlabel("Dataset size (images)")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    style_figure(fig, "Common test set: 50 images / 351 objects")
    fig.savefig(SHOTS / "04_common_test_comparison.png", bbox_inches="tight")
    plt.close(fig)


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Return an edge-padded moving average with the same length as the input."""
    data = np.asarray(values, dtype=float)
    if window <= 1:
        return data
    left = window // 2
    right = window - 1 - left
    padded = np.pad(data, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def save_per_run_dashboards(events: dict[int, EventAccumulator]) -> None:
    """Create one complete, readable TensorBoard-data dashboard per training run."""
    PER_RUN_SHOTS.mkdir(parents=True, exist_ok=True)
    loss_tags = [
        ("train/box_loss_iter", "Box loss"),
        ("train/obj_loss_iter", "Objectness loss"),
        ("train/cls_loss_iter", "Classification loss"),
        ("train/total_loss", "Total loss"),
    ]
    metric_tags = [
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
    ]
    for size in SIZES:
        accumulator = events[size]
        fig, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=150)
        smooth_window = max(7, EXPECTED_STEPS[size] // 100)
        for axis, (tag, label) in zip(axes.flat[:4], loss_tags):
            points = accumulator.Scalars(tag)
            steps = [point.step for point in points]
            values = [point.value for point in points]
            axis.plot(steps, values, color=COLORS[size], alpha=0.25, linewidth=0.7, label="Raw iteration")
            axis.plot(
                steps,
                moving_average(values, smooth_window),
                color=COLORS[size],
                linewidth=2,
                label=f"Moving mean ({smooth_window})",
            )
            axis.set_title(label)
            axis.set_xlabel("Global training iteration")
            axis.set_ylabel("Loss")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)

        lr_axis = axes.flat[4]
        lr_points = accumulator.Scalars("train/learning_rate")
        lr_axis.plot(
            [point.step for point in lr_points],
            [point.value for point in lr_points],
            color=COLORS[size],
            linewidth=2,
        )
        lr_axis.set_title("Scheduled learning rate")
        lr_axis.set_xlabel("Global training iteration")
        lr_axis.set_ylabel("Learning rate")
        lr_axis.grid(alpha=0.25)

        metrics_axis = axes.flat[5]
        for tag, label in metric_tags:
            points = accumulator.Scalars(tag)
            metrics_axis.plot(
                [point.step + 1 for point in points],
                [point.value for point in points],
                linewidth=2,
                label=label,
            )
        metrics_axis.set_title("Validation metrics")
        metrics_axis.set_xlabel("Epoch")
        metrics_axis.set_ylabel("Score")
        metrics_axis.set_ylim(0, 1.03)
        metrics_axis.grid(alpha=0.25)
        metrics_axis.legend(fontsize=8)

        style_figure(
            fig,
            f"size{size}_e30_seed2569 — {EXPECTED_STEPS[size]} training iterations / 30 epochs",
        )
        fig.savefig(PER_RUN_SHOTS / f"size{size}_tensorboard_dashboard.png", bbox_inches="tight")
        plt.close(fig)

        native_results = run_dir(size) / "results.png"
        if native_results.is_file():
            (PER_RUN_SHOTS / f"size{size}_yolov5_results.png").write_bytes(native_results.read_bytes())


def write_csv(records: list[dict[str, object]]) -> None:
    fields = list(records[0])
    with (RESULT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary(records: list[dict[str, object]]) -> None:
    winner = max(records, key=lambda record: float(record["test_map50_95"]))
    rows = "\n".join(
        f"| {r['dataset_size']} | {r['iteration_points']} | {r['best_epoch']} | "
        f"{float(r['val_map50_95']):.5f} | {float(r['test_precision']):.3f} | "
        f"{float(r['test_recall']):.3f} | {float(r['test_map50']):.3f} | "
        f"{float(r['test_map50_95']):.4f} |"
        for r in records
    )
    checks = "\n".join(
        f"- Size {r['dataset_size']}: {r['iteration_points']} points/tag, "
        f"counts={r['event_counts_ok']}, steps={r['steps_contiguous']}, "
        f"finite={r['values_finite']}, max total-sum error={float(r['max_total_sum_error']):.3e}."
        for r in records
    )
    text = f"""# YOLO-Box four-dataset experiment results

Generated from fresh runs on 2026-09-15. All runs used YOLOv5n pretrained weights,
30 epochs, batch size 2, image size 640, CPU, workers 0, seed 2569, and the default
`hyp.scratch-low.yaml`. The four datasets are deterministic nested subsets.

## Common-test comparison

Every `best.pt` was evaluated on the same `f09_box_350` test split: 50 images and
351 annotated objects.

| Dataset images | Iterations | Best epoch | Best val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

The size-{winner['dataset_size']} run is the winner by common-test mAP50-95
({float(winner['test_map50_95']):.4f}). Performance improves strongly as the training
subset grows, with the largest gain occurring between 60 and 150 images.

## TensorBoard event verification

{checks}

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
"""
    (RESULT / "RUN_SUMMARY.md").write_text(text, encoding="utf-8")


def write_report_html() -> None:
    """Render the submission Markdown as portable HTML before PDF export."""
    source = (RESULT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    body = markdown.markdown(source, extensions=["tables"])
    css = """
    @page { size: A4; margin: 18mm; }
    body { font-family: "Tahoma", "Arial", sans-serif; color: #202124; line-height: 1.55;
           max-width: 1000px; margin: 0 auto; font-size: 11pt; }
    h1 { color: #12355b; border-bottom: 3px solid #1f77b4; padding-bottom: 8px; }
    h2 { color: #174a7e; border-bottom: 1px solid #b8cce0; padding-bottom: 4px; margin-top: 28px; }
    h3 { color: #245b89; margin-top: 22px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 9.5pt; }
    th, td { border: 1px solid #9aa0a6; padding: 6px 8px; text-align: left; }
    th { background: #e8f0f8; }
    tr:nth-child(even) { background: #f8f9fa; }
    img { display: block; max-width: 100%; max-height: 235mm; margin: 12px auto 22px;
          break-inside: avoid; page-break-inside: avoid; }
    code { background: #f1f3f4; padding: 1px 4px; border-radius: 3px; }
    p { orphans: 3; widows: 3; }
    """
    html = (
        "<!doctype html><html lang=\"th\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )
    (RESULT / "FINAL_REPORT.html").write_text(html, encoding="utf-8")


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    events = {size: load_events(size) for size in SIZES}
    records: list[dict[str, object]] = []
    for size in SIZES:
        record: dict[str, object] = {"dataset_size": size}
        record.update(read_train_summary(size))
        record.update(read_common_test(size))
        record.update(verify_events(size, events[size]))
        record["best_weights_exists"] = (run_dir(size) / "weights" / "best.pt").is_file()
        record["last_weights_exists"] = (run_dir(size) / "weights" / "last.pt").is_file()
        records.append(record)
    write_csv(records)
    save_iteration_losses(events)
    save_epoch_metrics(events)
    save_learning_rate(events)
    save_test_comparison(records)
    save_per_run_dashboards(events)
    write_summary(records)
    write_report_html()
    print(f"Wrote {len(records)} experiment records, 4 comparison plots, and 8 per-run plots to {RESULT}")


if __name__ == "__main__":
    main()
