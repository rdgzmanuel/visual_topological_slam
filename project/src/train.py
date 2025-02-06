# deep learning libraries
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# other libraries
from tqdm.auto import tqdm

# own modules
from src.models import CNNExtractor
from src.utils import (
    load_cold_data,
    save_model,
    set_seed,
    Accuracy
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
SEQ_DATA_PATH: str = "seq_data"
DATA_PATH: str = "data"
NUMBER_OF_CLASSES: int = 12

def main() -> None:
    """
    This function is the main program for the training.
    """

    # hyperparameters
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 64
    dropout: float = 0.5

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_cold_data(SEQ_DATA_PATH, DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = "model_1"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = CNNExtractor(output_size=NUMBER_OF_CLASSES, dropout=dropout).to(device)
    
    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

        print(f"Epoch {epoch} finished.")

    # save model
    save_model(model, name)

    return None


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []
    accuracy: Accuracy = Accuracy()
    
    model.train()

    for images, labels in train_data:
        images = images.to(device)
        labels = labels.to(device)

        outputs: torch.Tensor = model(images)
        loss_value: torch.nn.Module = loss(outputs, labels.long())
        accuracy.update(outputs, labels.long())
        accuracy_value: float = accuracy.compute()

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        accuracies.append(accuracy_value)
        
    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """
    
    model.eval()

    with torch.no_grad():
        losses: list[float] = []
        accuracies: list[float] = []
        accuracy: Accuracy = Accuracy()

        for images, labels in val_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs: torch.Tensor = model(images)
            loss_value: torch.nn.Module = loss(outputs, labels.long())
            accuracy.update(outputs, labels.long())
            accuracy_value: float = accuracy.compute()

            losses.append(loss_value.item())
            accuracies.append(accuracy_value)
        
        writer.add_scalar("val/loss", np.mean(losses), epoch)
        writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)


if __name__ == "__main__":
    main()
