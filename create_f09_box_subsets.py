"""Create deterministic nested YOLO dataset subsets from datasets/f09_box."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasets" / "f09_box"
SEED = 2569
SPLIT_COUNTS = {
    30: {"train": 22, "val": 4, "test": 4},
    60: {"train": 42, "val": 9, "test": 9},
    150: {"train": 108, "val": 21, "test": 21},
    350: {"train": 250, "val": 50, "test": 50},
}


def load_manifest() -> dict[str, list[dict[str, str]]]:
    """Load and deterministically shuffle source records within each split."""
    with (SOURCE / "manifest.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    by_split = {split: [] for split in ("train", "val", "test")}
    for record in records:
        by_split[record["split"]].append(record)

    for index, split in enumerate(("train", "val", "test")):
        by_split[split].sort(key=lambda row: row["image"])
        random.Random(SEED + index).shuffle(by_split[split])
    return by_split


def validate_pair(split: str, image_name: str) -> tuple[Path, Path, int]:
    """Validate one source image-label pair and return its object count."""
    image_path = SOURCE / "images" / split / image_name
    label_path = SOURCE / "labels" / split / f"{Path(image_name).stem}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing pair for {split}/{image_name}")

    object_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        values = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Invalid coordinates at {label_path}:{line_number}")
        object_count += 1
    return image_path, label_path, object_count


def create_subset(size: int, by_split: dict[str, list[dict[str, str]]]) -> None:
    """Create one physical YOLO dataset folder with images, labels, and metadata."""
    destination = ROOT / "datasets" / f"f09_box_{size}"
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    selected: list[dict[str, str]] = []
    object_counts = {"train": 0, "val": 0, "test": 0}
    for split, count in SPLIT_COUNTS[size].items():
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)
        for record in by_split[split][:count]:
            image_path, label_path, objects = validate_pair(split, record["image"])
            shutil.copy2(image_path, destination / "images" / split / image_path.name)
            shutil.copy2(label_path, destination / "labels" / split / label_path.name)
            selected.append(record)
            object_counts[split] += objects

    selected.sort(key=lambda row: (("train", "val", "test").index(row["split"]), row["image"]))
    with (destination / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    (destination / "data.yaml").write_text(
        f"path: ../datasets/f09_box_{size}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: lane\n"
        "  1: track-line\n"
        "  2: sideway\n",
        encoding="utf-8",
    )
    split_text = ", ".join(f"{split} {count}" for split, count in SPLIT_COUNTS[size].items())
    object_text = ", ".join(f"{split} {count}" for split, count in object_counts.items())
    (destination / "README.md").write_text(
        f"# F09 YOLO Box — {size} images\n\n"
        f"Deterministic nested subset of `datasets/f09_box` using seed `{SEED}`.\n\n"
        f"- Images: {split_text}\n"
        f"- Objects: {object_text}\n"
        "- Classes: `lane`, `track-line`, `sideway`\n"
        "- Smaller datasets are contained in larger datasets within each split.\n",
        encoding="utf-8",
    )
    print(f"created {destination.name}: {split_text}; objects: {object_text}")


def main() -> None:
    """Create all requested subsets after validating source capacity and nesting."""
    by_split = load_manifest()
    for size, counts in SPLIT_COUNTS.items():
        for split, count in counts.items():
            if len(by_split[split]) < count:
                raise ValueError(f"Source split {split} has {len(by_split[split])} records; {count} required")
        create_subset(size, by_split)


if __name__ == "__main__":
    main()
