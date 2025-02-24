import os
import random
import torch
import tarfile
import requests
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.jit import RecursiveScriptModule
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



class COLDataset(Dataset):
    """
    This class is the COLD Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of COLDataset.

        Args:
            path: path of the dataset.
        """

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        labels_correspondence: dict[str, int] = {
            "CR": 0,
            "2PO": 1,
            "RL": 2,
            "TL": 3,
            "TR": 4,
            "LO": 5,
            "1PO": 6,
            "KT": 7,
            "CNR": 8,
            "PA": 9,
            "LAB": 10,
            "ST": 11
        }

        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        for image in os.listdir(path):
            image_splitted: list[str] = image[:-5].split("_")
            label_name: str = image_splitted[-1]

            image_path: str = os.path.join(path, image)
            open_image = Image.open(image_path).convert("RGB")
            tensor_image: torch.Tensor = transform(open_image)
            self.images.append(tensor_image)
            self.labels.append(labels_correspondence[label_name])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image: torch.Tensor = self.images[index]
        label: int = self.labels[index]

        return image, label


def load_cold_data(
    seq_data_path: str, data_path: str, batch_size: int = 128, num_workers: int = 4, train: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function prepares three Dataloaders, one for train data, one for val data and
    other for testing data for COLD dataset.

    Args:
        seq_data_path (str): path to the sequencies data.
        data_path (str): path where images will be stored.
        batch_size (int, optional): size of batch. Defaults to 128.
        num_workers (int, optional): number of workers for dataloaders. Defaults to 4.
        train (bool, optional): whether data is for training or testing. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: three dataloaders (train, val, test)
    """

    # if not os.path.isdir(f"{seq_data_path}"):
        # os.makedirs(f"{seq_data_path}")
    # download_cold_data(seq_data_path)
    
    # if not os.path.isdir(f"{data_path}"):
        # os.makedirs(f"{data_psath}")
    # prepare_data(seq_data_path, data_path)

    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    if train:
        train_dataset: Dataset = COLDataset(f"{data_path}/train")
        val_dataset: Dataset
        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])

        train_dataloader: DataLoader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_dataloader: DataLoader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    else:
        test_dataset: Dataset = COLDataset(f"{data_path}/test")
        
        test_dataloader: DataLoader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    return train_dataloader, val_dataloader, test_dataloader


def download_cold_data(path: str) -> None:
    """
    Downloads and extracts the COLD dataset, keeping only specific subfolders inside each sequence folder.

    Args:
        path (str): path to save the dataset.
    """
    
    base_url: str = "https://www.cas.kth.se/COLD/db/"
    labs: list[str] = ["cold-freiburg", "cold-ljubljana", "cold-saarbruecken"]
    parts: list[str] = ["part_a", "part_b"]
    weather_conditions: list[str] = ["cloudy1", "cloudy2", "cloudy3", "cloudy4", "cloudy5",
                                     "sunny1", "sunny2", "sunny3", "sunny4",
                                     "night1", "night2", "night3"]

    os.makedirs(path, exist_ok=True)

    keep_folders: list[str] = ["localization", "odom_scans", "std_cam"]

    for lab in labs:
        for part in parts:
            for seq_num in range(1, 5):
                for weather in weather_conditions:
                    seq_name: str = f"seq{seq_num}_{weather}"
                    save_name: str = f"{lab}_{part}_seq{seq_num}_{weather}"
                    url: str = f"{base_url}{lab}/{part}/{seq_name}.tar"
                    target_tar_path: str = os.path.join(path, f"{seq_name}.tar")
                    extracted_seq_path: str = os.path.join(path, save_name)

                    if os.path.exists(extracted_seq_path):
                        print(f"Skipping {save_name}, already exists.")
                        continue

                    print(f"Downloading {save_name} from {url}...")
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(target_tar_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                    else:
                        print(f"Failed to download {save_name}. HTTP status code: {response.status_code}")
                        continue

                    print(f"Extracting {save_name}...")
                    with tarfile.open(target_tar_path, "r:") as tar:
                        tar.extractall(path=extracted_seq_path)

                    extracted_contents = os.listdir(extracted_seq_path)
                    if len(extracted_contents) == 1 and os.path.isdir(os.path.join(extracted_seq_path, extracted_contents[0])):
                        nested_folder = os.path.join(extracted_seq_path, extracted_contents[0])
                        print(f"Moving contents from nested folder {nested_folder} to {extracted_seq_path}...")

                        for item in os.listdir(nested_folder):
                            shutil.move(os.path.join(nested_folder, item), extracted_seq_path)

                        os.rmdir(nested_folder)

                    if os.path.exists(extracted_seq_path):
                        for subfolder in os.listdir(extracted_seq_path):
                            subfolder_path = os.path.join(extracted_seq_path, subfolder)
                            if os.path.isdir(subfolder_path) and subfolder not in keep_folders:
                                print(f"Removing {subfolder_path}...")
                                shutil.rmtree(subfolder_path)
                    else:
                        print(f"Expected sequence folder not found at: {extracted_seq_path}")

                    os.remove(target_tar_path)

    print("Dataset downloaded, extracted and filtered successfully.")
    return None


def prepare_data(seq_data_path: str, final_data_path: str) -> None:
    """
    Copies images to a new folder while preserving their original names and appending class labels.
    77 sequences in total: 64 for training and leaving 13 of them for testing (83-17):
    Only testing with std path, not ext
    - 32 Saarbruecken: 5 for testing
    - 26 Freiburg: 5 for testing
    - 19 Ljubljana: 3 for testing

    Args:
        seq_data_path (str): path of the sequencies data
        final_data_path (str): path where images will be stored
    """
    sequences: str = os.listdir(seq_data_path)
    
    test_sequences: set[str] = {
        "cold-saarbruecken_part_a_seq1_cloudy1",
        "cold-saarbruecken_part_a_seq1_night1",
        "cold-saarbruecken_part_b_seq3_cloudy1",
        "cold-saarbruecken_part_b_seq3_night1",
        "cold-freiburg_part_a_seq1_cloudy1",
        "cold-freiburg_part_a_seq1_night1",
        "cold-freiburg_part_a_seq1_sunny1",
        "cold-freiburg_part_b_seq3_cloudy1",
        "cold-freiburg_part_b_seq3_sunny1",
        "cold-ljubljana_part_a_seq1_cloudy1",
        "cold-ljubljana_part_a_seq1_night1",
        "cold-ljubljana_part_a_seq1_sunny1"
    }

    classes: list[str] = [
        "CR",
        "2PO",
        "RL",
        "TL",
        "TR",
        "LO",
        "1PO",
        "KT",
        "CNR",
        "PA",
        "LAB",
        "ST"
    ]

    places_file_name: str = "localization/places.lst"
    camera_folder: str = "std_cam"

    for sequence in sequences:
        if sequence.startswith("."):
            continue

        sequence_path: str = os.path.join(seq_data_path, sequence)
        places_path: str = os.path.join(sequence_path, places_file_name)
        pictures_path: str = os.path.join(sequence_path, camera_folder)

        picture_to_class: dict[str: str] = {}
        with open(places_path, "r") as file:
            for line in file:
                parts: list[str] = line.strip().split()
                if len(parts) == 2:
                    concrete_place: str = parts[1]
                    for class_ in classes:
                        if class_ in concrete_place:
                            picture_to_class[parts[0]] = class_
                            break

        train_test: str = "test" if sequence in test_sequences else "train"
        data_folder_path: str = os.path.join(final_data_path, train_test)
        os.makedirs(data_folder_path, exist_ok=True)

        for picture in os.listdir(pictures_path):
            if picture in picture_to_class:
                picture_class: str = picture_to_class[picture]
                original_name, ext = os.path.splitext(picture)
                new_name: str = f"{original_name}_{picture_class}{ext}"

                dest_path: str = os.path.join(data_folder_path, new_name)
                if not os.path.exists(dest_path):
                    src_path: str = os.path.join(pictures_path, picture)
                    shutil.copy(src_path, dest_path)
        
        print(f"Sequence {sequence} completed.")
    
    return None



class Accuracy:
    """
    This class tracks the accuracy of predictions.

    Attributes:
        correct (int): number of correct predictions.
        total (int): total number of examples evaluated.
    """

    def __init__(self) -> None:
        """Initializes correct and total counts to zero."""
        self.correct: int = 0
        self.total: int= 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the count of correct and total predictions.

        Args:
            logits (torch.Tensor): model outputs of shape [batch, num_classes].
            labels (torch.Tensor): ground truth labels of shape [batch].
        """
        predictions: torch.Tensor = logits.argmax(dim=1)
        self.correct += predictions.eq(labels).sum().item()
        self.total += labels.size(0)

    def compute(self) -> float:
        """
        Computes the accuracy.

        Returns:
            float: accuracy as a value between 0 and 1.
        """
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        """Resets the correct and total counts to zero."""
        self.correct = 0
        self.total = 0


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name (str): name of the model to load.

    Returns:
        RecursiveScriptModule: model in torchscript.
    """

    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None

def load_tensorboard_scalars(model_dir: str, metric: str) -> tuple[list[int], list[float]]:
    """
    Loads scalar data from TensorBoard log files.

    Args:
        model_dir (str): path to the model's log directory.
        metric (str): name of the scalar metric to extract.

    Returns:
        tuple[list[int], list[float]]: steps and corresponding values if metric exists, otherwise None.
    """
    event_acc: EventAccumulator = EventAccumulator(model_dir)
    event_acc.Reload()
    
    if metric not in event_acc.Tags()["scalars"]:
        return None

    events = event_acc.Scalars(metric)
    steps: list[int] = [e.step for e in events]
    values: list[float] = [e.value for e in events]
    
    return steps, values


def plot_images() -> None:
    images_path: str = "comparison"
    models: list[str] = ["model_8", "model_13", "cnn_avg_1", "ae_avg_4"]

    model_names: dict[str, str] = {
        "model_8": "m8_cnn",
        "model_13": "m13_ae_d",
        "cnn_avg_1": "cnn_avg_1",
        "ae_avg_4": "ae_avg_4",
    }

    colors: list[str] = ["#007acc", "#ff7700", "#33cc33", "#cc33cc"]

    plot_charts(images_path, models, model_names, colors)

    return None
    

def plot_charts(images_path: str, models: list[str], model_names: dict[str, str], colors: list[str]) -> None:
    log_dir: str = "runs"
    image_dir: str = os.path.join("images", images_path)
    metrics: list[str] = ["train/loss", "train/accuracy", "val/loss", "val/accuracy"]

    os.makedirs(image_dir, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    for metric in metrics:
        plt.figure(figsize=(8, 6))

        for i, model in enumerate(models):
            model_path: str = os.path.join(log_dir, model)
            data: tuple[list[int], list[float]] = load_tensorboard_scalars(model_path, metric)
            
            if data:
                steps, values = data
                plt.plot(steps, values, label=model_names[model], linewidth=2.5, markersize=5, color=colors[i])

        plt.title(metric.replace("/", " - ").title(), fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        image_path: str = os.path.join(image_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    path: str = "seq_data"
    data_path: str = "data"
    # download_cold_data(path)
    # prepare_data(path, data_path)
    # load_cold_data(data_path)
    plot_images()