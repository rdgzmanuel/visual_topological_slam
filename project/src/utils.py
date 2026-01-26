import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.jit import RecursiveScriptModule


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    Saves a model in the 'models' folder as torch.jit.
    Creates the 'models' folder if it doesn't exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # Create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    Loads a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        RecursiveScriptModule: model in torchscript.
    """
    model_path: str = f"/workspace/project/models/{name}.pt"

    if not os.path.exists(model_path):
        model_path = f"models/{name}.pt"

    model: RecursiveScriptModule = torch.jit.load(model_path, map_location="cpu")

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
