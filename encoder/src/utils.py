import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    from src.config import DATA_PATH, IMAGES_PATH
    from src.data import COLDEvaluationDataset
except ImportError:
    from config import DATA_PATH, IMAGES_PATH
    from data import COLDEvaluationDataset


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
            "dino_model": getattr(model, "dino_model", "dinov2_vits14"),
        },
        filepath,
    )

    print(f"Model saved to {filepath}")


def load_model(
    name: str, load_dir: str = "models", device: str = "cuda"
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
    try:
        from src.models import VisualEncoderDINO
    except ImportError:
        from models import VisualEncoderDINO

    encoder_dir: Path = Path(__file__).resolve().parent.parent  # points to encoder/
    filepath: Path = encoder_dir / load_dir / f"{name}.pth"

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Extract model config
    embedding_dim = checkpoint.get("embedding_dim", 128)
    dino_model = checkpoint.get("dino_model", "dinov2_vits14")

    # Create model with same architecture
    model = VisualEncoderDINO(
        embedding_dim=embedding_dim,
        dino_model=dino_model,
    )

    # Load state dict
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
    train_losses: list[float], val_losses: list[float], name: str
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
    plt.style.use("seaborn-v0_8-darkgrid")

    epochs = np.arange(1, len(train_losses) + 1)

    # Create figure with higher DPI for publication quality
    _, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Plot with professional styling
    ax.plot(
        epochs,
        train_losses,
        color="#2E86AB",
        linewidth=2.5,
        label="Training Loss",
        marker="o",
        markersize=4,
        markevery=max(1, len(epochs) // 20),
    )

    ax.plot(
        epochs,
        val_losses,
        color="#A23B72",
        linewidth=2.5,
        label="Validation Loss",
        marker="s",
        markersize=4,
        markevery=max(1, len(epochs) // 20),
    )

    # Customize axes
    ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax.set_title(
        "Training and Validation Loss Curves", fontsize=16, fontweight="bold", pad=20
    )

    # Customize legend
    ax.legend(fontsize=12, frameon=True, shadow=True, loc="best", fancybox=True)

    # Customize grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # Customize tick parameters
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add best validation loss annotation
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    ax.annotate(
        f"Best: {best_val_loss:.4f}",
        xy=(best_val_idx + 1, best_val_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=1.5
        ),
    )

    # Tight layout for better spacing
    plt.tight_layout()

    # Save with high quality
    plt.savefig(
        IMAGES_PATH / f"{name}_loss.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    print(f"\nTraining curves saved to {IMAGES_PATH / f'{name}_loss.png'}")


def visualize_embeddings_umap(
    model_name: str,
    num_samples: int = 500,
    load_dir: str = "models",
    device: str = "cuda",
    random_state: int = 42,
) -> None:
    """
    Visualize embeddings using UMAP for a given trained model.

    Randomly samples images from the test set, extracts embeddings using the model,
    and creates a 2D UMAP visualization colored by class.

    Args:
        model_name: name of the saved model (without .pth extension).
        num_samples: number of samples to visualize (will be capped by dataset size).
        load_dir: directory containing saved models.
        device: device to run inference on.
        random_state: random seed for reproducibility.
    """

    # Load the model
    model = load_model(model_name, load_dir=load_dir, device=device)
    model.eval()

    # Load test dataset
    test_dataset = COLDEvaluationDataset(f"{DATA_PATH}/test")

    # Cap samples to dataset size
    num_samples = min(num_samples, len(test_dataset))

    # Randomly sample indices
    np.random.seed(random_state)
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)

    # Extract embeddings
    embeddings: list[np.ndarray] = []
    labels: list[int] = []

    print(f"Extracting embeddings for {num_samples} samples...")
    with torch.no_grad():
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            embedding = model.extract_features(image)
            embeddings.append(embedding.cpu().numpy().squeeze())
            labels.append(label)

    embeddings_array = np.stack(embeddings)
    labels_array = np.array(labels)

    # Apply UMAP
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",  # Good for normalized embeddings
        random_state=random_state,
    )
    embeddings_2d = reducer.fit_transform(embeddings_array)

    # Create label name mapping (inverse of labels_correspondence)
    label_names: dict[int, str] = {
        0: "CR",
        1: "2PO",
        2: "RL",
        3: "TL",
        4: "TR",
        5: "LO",
        6: "1PO",
        7: "KT",
        8: "CNR",
        9: "PA",
        10: "LAB",
        11: "ST",
    }

    # Plot
    IMAGES_PATH.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    _, ax = plt.subplots(figsize=(14, 10), dpi=300)

    # Get unique labels present in the sample
    unique_labels = np.unique(labels_array)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels_array == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=label_names.get(label, f"Class {label}"),
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=14, fontweight="bold")
    ax.set_title(
        f"UMAP Visualization of Embeddings\nModel: {model_name}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.legend(
        fontsize=10,
        frameon=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Classes",
        title_fontsize=12,
    )

    ax.tick_params(axis="both", which="major", labelsize=11)

    plt.tight_layout()

    save_path = IMAGES_PATH / f"{model_name}_umap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"UMAP visualization saved to {save_path}")


if __name__ == "__main__":
    loss: str = "contrastive"  # triplet or contrastive
    assert loss in ["triplet", "contrastive"]

    visualize_embeddings_umap(
        model_name=f"visual_encoder_dino_{loss}_dim128_best",
        num_samples=500,
        device="cuda",
    )
