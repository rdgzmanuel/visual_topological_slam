import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.jit import RecursiveScriptModule
from pathlib import Path

SEQ_DATA_PATH: str = "seq_data"
DATA_PATH: str = "data"
IMAGES_PATH: Path = Path("images")


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
    train_losses: list[float],
    val_losses: list[float],
    name: str
) -> None:
    """
    Plot and save professional training curves for loss.

    Args:
        train_losses: list of training losses per epoch.
        val_losses: list of validation losses per epoch.
        name: name for the saved plot.
    """
    IMAGES_PATH.mkdir(exist_ok=True)
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create figure with higher DPI for publication quality
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Plot with professional styling
    ax.plot(epochs, train_losses, 
            color='#2E86AB', 
            linewidth=2.5, 
            label='Training Loss',
            marker='o',
            markersize=4,
            markevery=max(1, len(epochs)//20))
    
    ax.plot(epochs, val_losses, 
            color='#A23B72', 
            linewidth=2.5, 
            label='Validation Loss',
            marker='s',
            markersize=4,
            markevery=max(1, len(epochs)//20))
    
    # Customize axes
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves', 
                 fontsize=16, 
                 fontweight='bold',
                 pad=20)
    
    # Customize legend
    ax.legend(fontsize=12, 
             frameon=True, 
             shadow=True,
             loc='best',
             fancybox=True)
    
    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add best validation loss annotation
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    ax.annotate(f'Best: {best_val_loss:.4f}',
                xy=(best_val_idx + 1, best_val_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='yellow', 
                         alpha=0.7),
                arrowprops=dict(arrowstyle='->', 
                               connectionstyle='arc3,rad=0',
                               color='black',
                               lw=1.5))
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(IMAGES_PATH / f"{name}_loss.png", 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    print(f"\nTraining curves saved to {IMAGES_PATH / f'{name}_loss.png'}")

if __name__ == "__main__":
    plot_training_curves(
        run_names=["visual_encoder_dino_triplet_dim128"],
        display_names={"visual_encoder_dino_triplet_dim128": "DINOv2 Triplet"},
    )
