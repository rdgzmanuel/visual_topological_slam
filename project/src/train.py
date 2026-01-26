import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.models import ContrastiveLoss, TripletLoss, VisualEncoderDINO
from src.utils import load_triplet_data, save_model, set_seed

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

set_seed(42)
torch.set_num_threads(8)

SEQ_DATA_PATH: str = "seq_data"
DATA_PATH: str = "data"


def main() -> None:
    """
    Main training loop for visual encoder with metric learning using DINOv2.
    """

    # Hyperparameters
    epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 32
    embedding_dim: int = 128
    dropout: float = 0.3
    margin: float = 0.3
    temperature: float = 0.07
    loss_type: str = "contrastive"  # "triplet" or "contrastive"
    dino_model: str = "dinov2_vits14"  # small, fast model

    # Learning rate schedule
    step_size: int = 20
    gamma: float = 0.5

    open("nohup.out", "w").close()

    # Load data as triplets (anchor, positive, negative)
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data = load_triplet_data(
        SEQ_DATA_PATH, DATA_PATH, batch_size=batch_size
    )

    name: str = f"visual_encoder_dino_{loss_type}_dim{embedding_dim}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    model: VisualEncoderDINO = VisualEncoderDINO(
        embedding_dim=embedding_dim, dropout=dropout, dino_model=dino_model
    ).to(device)

    if loss_type == "triplet":
        criterion = TripletLoss(margin=margin)
    else:
        criterion = ContrastiveLoss(temperature=temperature)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": lr * 0.1,
            },
            {"params": model.projection_head.parameters(), "lr": lr},
        ],
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    best_val_loss: float = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = train_step(
            model, train_data, criterion, optimizer, writer, epoch, device, loss_type
        )

        val_loss = val_step(
            model, val_data, criterion, writer, epoch, device, loss_type
        )

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f"{name}_best")
            print(
                f"\nNew best model saved at epoch {epoch} with val_loss: {val_loss:.4f}"
            )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

    save_model(model, name)
    writer.close()

    return None


def train_step(
    model: VisualEncoderDINO,
    train_data: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    loss_type: str,
) -> float:
    """
    Training step with metric learning.

    Args:
        model: visual encoder model.
        train_data: dataloader providing triplets.
        criterion: loss function (triplet or contrastive).
        optimizer: optimizer.
        writer: tensorboard writer.
        epoch: current epoch.
        device: device.
        loss_type: type of loss being used.

    Returns:
        average training loss.
    """
    model.train()
    losses: list[float] = []

    for batch in tqdm(train_data, desc="Training", leave=False):
        if loss_type == "triplet":
            anchor_imgs, positive_imgs, negative_imgs = batch
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)

            # Forward pass
            anchor_emb = model(anchor_imgs)
            positive_emb = model(positive_imgs)
            negative_emb = model(negative_imgs)

            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
        else:  # contrastive
            anchor_imgs, positive_imgs, negative_imgs_list = batch
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs_list = negative_imgs_list.to(
                device
            )  # [batch, num_neg, C, H, W]

            # Forward pass
            anchor_emb = model(anchor_imgs)
            positive_emb = model(positive_imgs)

            # Process negatives
            batch_size, num_neg = negative_imgs_list.shape[:2]
            negative_imgs_flat = negative_imgs_list.view(
                -1, *negative_imgs_list.shape[2:]
            )
            negative_emb_flat = model(negative_imgs_flat)
            negative_emb = negative_emb_flat.view(batch_size, num_neg, -1)

            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    writer.add_scalar("train/loss", avg_loss, epoch)

    return avg_loss


def val_step(
    model: VisualEncoderDINO,
    val_data: DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    loss_type: str,
) -> float:
    """
    Validation step.

    Args:
        model: visual encoder model.
        val_data: validation dataloader.
        criterion: loss function.
        writer: tensorboard writer.
        epoch: current epoch.
        device: device.
        loss_type: type of loss being used.

    Returns:
        average validation loss.
    """
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for batch in tqdm(val_data, desc="Validation", leave=False):
            if loss_type == "triplet":
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_imgs = anchor_imgs.to(device)
                positive_imgs = positive_imgs.to(device)
                negative_imgs = negative_imgs.to(device)

                anchor_emb = model(anchor_imgs)
                positive_emb = model(positive_imgs)
                negative_emb = model(negative_imgs)

                loss = criterion(anchor_emb, positive_emb, negative_emb)
            else:
                anchor_imgs, positive_imgs, negative_imgs_list = batch
                anchor_imgs = anchor_imgs.to(device)
                positive_imgs = positive_imgs.to(device)
                negative_imgs_list = negative_imgs_list.to(device)

                anchor_emb = model(anchor_imgs)
                positive_emb = model(positive_imgs)

                batch_size, num_neg = negative_imgs_list.shape[:2]
                negative_imgs_flat = negative_imgs_list.view(
                    -1, *negative_imgs_list.shape[2:]
                )
                negative_emb_flat = model(negative_imgs_flat)
                negative_emb = negative_emb_flat.view(batch_size, num_neg, -1)

                loss = criterion(anchor_emb, positive_emb, negative_emb)

            losses.append(loss.item())

    avg_loss = np.mean(losses)
    writer.add_scalar("val/loss", avg_loss, epoch)

    return avg_loss


if __name__ == "__main__":
    main()
