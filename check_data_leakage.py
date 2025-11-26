"""Diagnostics for spotting potential train/validation data leakage in OUHANDS."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import get_config
from ouhands_loader import OuhandsDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check for overlap between training and validation splits.")
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Hash image bytes to catch duplicates even when file names differ (slower).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a detailed leakage report (JSON).",
    )
    return parser.parse_args()


def instantiate_dataset(split: str, config) -> OuhandsDS:
    return OuhandsDS(
        root_dir=config.dataset.root_dir,
        split=split,
        transform=None,
        target_transform=None,
        return_paths=False,
        class_subset=None,
        train_subset_ratio=config.dataset.train_subset_ratio if split == "train" else 1.0,
        random_seed=config.dataset.random_seed,
    )


def collect_paths(dataset: OuhandsDS) -> Tuple[List[Path], List[str]]:
    paths = []
    classes = []
    for image_path, class_idx in dataset.samples:
        paths.append(image_path)
        classes.append(dataset.get_class_name(class_idx))
    return paths, classes


def resolve_paths(paths: Iterable[Path]) -> List[Path]:
    return [Path(p).resolve() for p in paths]


def check_name_overlap(train_paths: List[Path], val_paths: List[Path]) -> Dict[str, List[str]]:
    overlap_by_name: Dict[str, List[str]] = defaultdict(list)
    train_map = defaultdict(list)
    val_map = defaultdict(list)

    for path in train_paths:
        train_map[path.name].append(str(path))
    for path in val_paths:
        val_map[path.name].append(str(path))

    shared_names = set(train_map) & set(val_map)
    for name in sorted(shared_names):
        overlap_by_name[name] = train_map[name] + val_map[name]
    return overlap_by_name


def check_path_overlap(train_paths: List[Path], val_paths: List[Path]) -> List[Tuple[str, str]]:
    train_set = {str(p) for p in train_paths}
    val_set = {str(p) for p in val_paths}
    shared = sorted(train_set & val_set)
    return [(p, p) for p in shared]


def check_hash_overlap(train_paths: List[Path], val_paths: List[Path]) -> Dict[str, Dict[str, List[str]]]:
    def digest_map(paths: Iterable[Path]) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = defaultdict(list)
        for path in paths:
            with path.open("rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            mapping[digest].append(str(path))
        return mapping

    train_hashes = digest_map(train_paths)
    val_hashes = digest_map(val_paths)

    shared_hashes = set(train_hashes) & set(val_hashes)
    report: Dict[str, Dict[str, List[str]]] = {}
    for digest in sorted(shared_hashes):
        report[digest] = {"train": train_hashes[digest], "validation": val_hashes[digest]}
    return report


def summarize(classes: List[str]) -> Counter:
    return Counter(classes)


def main() -> None:
    args = parse_args()
    config = get_config()

    train_ds = instantiate_dataset("train", config)
    val_ds = instantiate_dataset("validation", config)

    train_paths, train_classes = collect_paths(train_ds)
    val_paths, val_classes = collect_paths(val_ds)

    resolved_train = resolve_paths(train_paths)
    resolved_val = resolve_paths(val_paths)

    by_path = check_path_overlap(resolved_train, resolved_val)
    by_name = check_name_overlap(resolved_train, resolved_val)

    hash_report = {}
    if args.hash_images:
        hash_report = check_hash_overlap(resolved_train, resolved_val)

    train_counts = summarize(train_classes)
    val_counts = summarize(val_classes)

    print("=== OUHANDS Split Diagnostics ===")
    print(f"Train samples: {len(train_paths)} | Validation samples: {len(val_paths)}")
    print(f"Unique train classes: {len(train_counts)} | Unique validation classes: {len(val_counts)}")
    print()

    print("Class distribution (train):")
    for cls, count in sorted(train_counts.items()):
        print(f"  {cls}: {count}")

    print("\nClass distribution (validation):")
    for cls, count in sorted(val_counts.items()):
        print(f"  {cls}: {count}")

    print("\nExact path overlap:")
    if by_path:
        for train_path, _ in by_path:
            print(f"  {train_path}")
    else:
        print("  None detected")

    print("\nFilename overlap (different directories):")
    if by_name:
        for name, occurrences in by_name.items():
            print(f"  {name}")
            for occ in occurrences:
                print(f"    {occ}")
    else:
        print("  None detected")

    if args.hash_images:
        print("\nContent overlap via SHA-256 hashes:")
        if hash_report:
            for digest, locations in hash_report.items():
                print(f"  hash={digest}")
                for split, paths in locations.items():
                    for path in paths:
                        print(f"    {split}: {path}")
        else:
            print("  None detected")

    if args.output:
        import json

        payload = {
            "train_samples": len(train_paths),
            "validation_samples": len(val_paths),
            "class_distribution_train": train_counts,
            "class_distribution_validation": val_counts,
            "path_overlap": by_path,
            "filename_overlap": by_name,
            "hash_overlap": hash_report,
        }

        # Counter is not JSON serializable; convert to dicts
        payload["class_distribution_train"] = dict(train_counts)
        payload["class_distribution_validation"] = dict(val_counts)

        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nDetailed report saved to {args.output}")


if __name__ == "__main__":
    main()
