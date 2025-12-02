"""Generate a t-SNE visualization for a single DINO checkpoint on OUHANDS."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config import ExperimentConfig, get_config
from evaluate import create_eval_dataloaders
from model import build_model
from utils import select_device


def collect_cls_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract CLS embeddings and labels from a loader."""
    model.eval()
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            batch_labels = batch[1]
            batch_paths = batch[-1] if isinstance(batch[-1], list) else None

            feats = model.forward_features(images)
            cls_tokens = feats[:, 0].detach().cpu().numpy()

            embeddings.append(cls_tokens)
            labels.extend(batch_labels.cpu().tolist())
            if batch_paths is not None:
                paths.extend(batch_paths)

            if max_samples is not None and len(labels) >= max_samples:
                break

    if not embeddings:
        raise RuntimeError("No embeddings collected; check that the loader is non-empty")

    embedding_array = np.concatenate(embeddings, axis=0)
    if max_samples is not None:
        max_samples = min(max_samples, embedding_array.shape[0], len(labels))
        embedding_array = embedding_array[:max_samples]
        labels = labels[:max_samples]
        paths = paths[:max_samples]

    return embedding_array, np.array(labels), paths


def run_tsne(embeddings: np.ndarray, seed: int, perplexity: float = 30.0) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate=200.0,
    )
    return tsne.fit_transform(embeddings)


def plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Iterable[str],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))

    cmap = plt.get_cmap("tab10")
    class_names = list(class_names)
    num_classes = len(class_names)

    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        if not np.any(mask):
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            color=cmap(cls_idx % cmap.N),
            label=class_names[cls_idx],
        )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _split_state_and_metadata(
    checkpoint_obj: object,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if isinstance(checkpoint_obj, Mapping):
        tensor_like = {k: v for k, v in checkpoint_obj.items() if isinstance(v, torch.Tensor)}

        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            nested = checkpoint_obj.get(key)
            if isinstance(nested, Mapping) and all(isinstance(v, torch.Tensor) for v in nested.values()):
                metadata = {k: v for k, v in checkpoint_obj.items() if k != key}
                return dict(nested), metadata

        if tensor_like and len(tensor_like) == len(checkpoint_obj):
            return dict(checkpoint_obj), {}

        if tensor_like:
            metadata = {k: v for k, v in checkpoint_obj.items() if k not in tensor_like}
            return tensor_like, metadata

    if isinstance(checkpoint_obj, torch.nn.Module):
        return checkpoint_obj.state_dict(), {}

    if isinstance(checkpoint_obj, (dict,)):
        return dict(checkpoint_obj), {}

    raise TypeError("Unsupported checkpoint format; expected a state dict or mapping container")


def load_checkpoint(checkpoint: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    loaded = torch.load(checkpoint, map_location=device)
    state_dict, metadata = _split_state_and_metadata(loaded)
    return _strip_module_prefix(state_dict), metadata


def load_weights(model: nn.Module, state_dict: Dict[str, torch.Tensor], checkpoint: Path) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing:
        print(f"Warning: missing keys while loading {checkpoint.name}: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys while loading {checkpoint.name}: {unexpected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a t-SNE plot for a DINO checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint to visualize")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max samples to use for t-SNE (keeps runtime manageable)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tsne_dino.png"),
        help="Where to save the resulting plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config: ExperimentConfig = get_config()
    device = select_device()

    cpu_device = torch.device("cpu")
    model, transforms = build_model(config, device)

    state_dict, _ = load_checkpoint(args.checkpoint, cpu_device)
    load_weights(model, state_dict, args.checkpoint)

    loaders = create_eval_dataloaders(config, transforms, device)
    loader = loaders[args.split]

    embeddings, labels, _ = collect_cls_embeddings(
        model,
        loader,
        device,
        max_samples=args.max_samples,
    )

    coords = run_tsne(embeddings, seed=args.seed, perplexity=args.perplexity)

    class_names = loader.dataset.classes  # type: ignore[attr-defined]
    plot_embeddings(
        coords,
        labels,
        class_names,
        title=args.checkpoint.stem,
        output_path=args.output,
    )
    print(f"Saved t-SNE visualization to {args.output}")


if __name__ == "__main__":
    main()
