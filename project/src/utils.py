import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from models import VisualEncoderDINO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def save_model(model: torch.nn.Module, name: str, save_dir: str = "models") -> None:
    """
    Save model state dict (not the entire model).

    Args:
        model: the model to save.
        name: name for the saved model.
        save_dir: directory to save models.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / f"{name}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "backbone_dim": model.backbone_dim,
            "dino_version": getattr(model, "dino_version", "v2"),
            "dino_model": getattr(model, "dino_model", "dinov2_vits14"),
        },
        filepath,
    )

    print(f"Model saved to {filepath}")


def load_model(
    name: str, load_dir: str = "models", device: str = "cpu"
) -> torch.nn.Module:
    """
    Load model from state dict.

    Args:
        name: name of the saved model.
        load_dir: directory containing saved models.
        device: device to load model on.

    Returns:
        loaded model.
    """
    project_root = Path(__file__).resolve().parents[1]  # src → project
    models_dir = project_root / load_dir

    filepath = models_dir / f"{name}.pth"

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    embedding_dim = checkpoint.get("embedding_dim", 128)
    dino_model = checkpoint.get("dino_model", "dinov2_vits14")

    model: VisualEncoderDINO = VisualEncoderDINO(
        embedding_dim=embedding_dim,
        dino_model=dino_model,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {filepath}")

    return model


def set_seed(seed: int) -> None:
    """
    Sets a seed and ensures deterministic behavior.

    Args:
        seed: seed number to fix randomness.
    """

    # Set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # Set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def load_tensorboard_scalars(
    model_dir: str, metric: str
) -> tuple[list[int], list[float]] | None:
    """
    Loads scalar data from TensorBoard log files.

    Args:
        model_dir: path to the model's log directory.
        metric: name of the scalar metric to extract.

    Returns:
        tuple of (steps, values) if metric exists, otherwise None.
    """
    event_acc: EventAccumulator = EventAccumulator(model_dir)
    event_acc.Reload()

    if metric not in event_acc.Tags()["scalars"]:
        return None

    events = event_acc.Scalars(metric)
    steps: list[int] = [e.step for e in events]
    values: list[float] = [e.value for e in events]

    return steps, values


def plot_training_curves(
    run_names: list[str],
    display_names: dict[str, str] | None = None,
    colors: list[str] | None = None,
    save_dir: str = "images/training_curves",
) -> None:
    """
    Plots training curves from TensorBoard logs.

    Args:
        run_names: list of run names (folder names in runs/).
        display_names: optional mapping of run names to display names.
        colors: optional list of colors for each run.
        save_dir: directory to save the plots.
    """
    log_dir: str = "runs"
    metrics: list[str] = ["train/loss", "val/loss"]

    os.makedirs(save_dir, exist_ok=True)

    if display_names is None:
        display_names = {name: name for name in run_names}

    if colors is None:
        colors = ["#007acc", "#ff7700", "#33cc33", "#cc33cc", "#ff0066", "#00cc99"]

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for i, run_name in enumerate(run_names):
            run_path: str = os.path.join(log_dir, run_name)
            data = load_tensorboard_scalars(run_path, metric)

            if data:
                steps, values = data
                plt.plot(
                    steps,
                    values,
                    label=display_names[run_name],
                    linewidth=2.5,
                    color=colors[i % len(colors)],
                )

        plt.title(metric.replace("/", " - ").title(), fontsize=16, fontweight="bold")
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        image_path: str = os.path.join(save_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {image_path}")


if __name__ == "__main__":
    plot_training_curves(
        run_names=["visual_encoder_dino_triplet_dim128"],
        display_names={"visual_encoder_dino_triplet_dim128": "DINOv2 Triplet"},
    )
