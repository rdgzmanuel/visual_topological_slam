import os
import random
import shutil
import tarfile
from collections import defaultdict

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class COLDTripletDataset(Dataset):
    """
    COLD Dataset for triplet learning.
    Returns (anchor, positive, negative) triplets for metric learning.
    """

    def __init__(self, path: str, augment: bool = True) -> None:
        """
        Constructor of COLDTripletDataset.

        Args:
            path: path of the dataset.
            augment: whether to apply data augmentation.
        """

        # Base transform (always applied)
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet stats for DINOv2
        ])

        # Augmentation transform (for anchors and positives during training)
        augment_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.base_transform = base_transform
        self.augment_transform = augment_transform if augment else base_transform
        self.augment = augment

        self.labels_correspondence: dict[str, int] = {
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
            "ST": 11,
        }

        # Store image paths and labels instead of loading all images into memory
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        # Group images by label for efficient triplet sampling
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)

        for image_name in os.listdir(path):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_splitted: list[str] = image_name[:-5].split("_")
            label_name: str = image_splitted[-1]

            if label_name not in self.labels_correspondence:
                continue

            image_path: str = os.path.join(path, image_name)
            label: int = self.labels_correspondence[label_name]

            idx = len(self.image_paths)
            self.image_paths.append(image_path)
            self.labels.append(label)
            self.label_to_indices[label].append(idx)

        # Verify we have multiple samples per class
        self.valid_labels: list[int] = [
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]

        if len(self.valid_labels) < 2:
            raise ValueError(
                f"Need at least 2 classes with 2+ samples each. Found {len(self.valid_labels)}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a triplet (anchor, positive, negative).

        Args:
            index: index of the anchor image.

        Returns:
            tuple of (anchor, positive, negative) tensors.
        """
        # Get anchor
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]
        anchor_img = Image.open(anchor_path).convert("RGB")

        # Get positive (same class, different image)
        positive_indices = [
            i for i in self.label_to_indices[anchor_label] if i != index
        ]
        if not positive_indices:
            # Fallback: use the same image (shouldn't happen if we have 2+ samples per class)
            positive_idx = index
        else:
            positive_idx = random.choice(positive_indices)

        positive_path = self.image_paths[positive_idx]
        positive_img = Image.open(positive_path).convert("RGB")

        # Get negative (different class)
        negative_labels = [
            label for label in self.valid_labels if label != anchor_label
        ]
        negative_label = random.choice(negative_labels)
        negative_idx = random.choice(self.label_to_indices[negative_label])

        negative_path = self.image_paths[negative_idx]
        negative_img = Image.open(negative_path).convert("RGB")

        # Apply transforms
        anchor_tensor = self.augment_transform(anchor_img)
        positive_tensor = self.augment_transform(positive_img)
        negative_tensor = self.base_transform(
            negative_img
        )  # No augmentation for negatives

        return anchor_tensor, positive_tensor, negative_tensor


class COLDEvaluationDataset(Dataset):
    """
    COLD Dataset for evaluation (no triplets, just images and labels).
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of COLDEvaluationDataset.

        Args:
            path: path of the dataset.
        """

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.labels_correspondence: dict[str, int] = {
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
            "ST": 11,
        }

        self.image_paths: list[str] = []
        self.labels: list[int] = []

        for image_name in os.listdir(path):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_splitted: list[str] = image_name[:-5].split("_")
            label_name: str = image_splitted[-1]

            if label_name not in self.labels_correspondence:
                continue

            image_path: str = os.path.join(path, image_name)
            self.image_paths.append(image_path)
            self.labels.append(self.labels_correspondence[label_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Returns image and label.

        Args:
            index: index of the image.

        Returns:
            tuple of (image tensor, label).
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, label


def load_triplet_data(
    seq_data_path: str,
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Prepares DataLoaders for triplet learning.

    Args:
        seq_data_path: path to the sequences data (not used, kept for compatibility).
        data_path: path where images are stored.
        batch_size: size of batch.
        num_workers: number of workers for dataloaders.

    Returns:
        tuple[DataLoader, DataLoader]: train and validation dataloaders.
    """

    train_dataset = COLDTripletDataset(f"{data_path}/train", augment=True)

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for triplet loss
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def load_evaluation_data(
    seq_data_path: str,
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Prepares DataLoaders for evaluation (place recognition).

    For evaluation, we treat test set as queries and train set as database.
    This simulates the real scenario: robot has seen training locations,
    and needs to recognize test locations.

    Args:
        seq_data_path: path to the sequences data (not used, kept for compatibility).
        data_path: path where images are stored.
        batch_size: size of batch.
        num_workers: number of workers for dataloaders.

    Returns:
        tuple[DataLoader, DataLoader]: query and database dataloaders.
    """

    # Query set: test images (robot trying to localize)
    query_dataset = COLDEvaluationDataset(f"{data_path}/test")

    # Database set: train images (known places)
    database_dataset = COLDEvaluationDataset(f"{data_path}/train")

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    database_dataloader = DataLoader(
        database_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return query_dataloader, database_dataloader


def download_cold_data(path: str) -> None:
    """
    Downloads and extracts the COLD dataset (Freiburg and Saarbrucken only).

    Args:
        path: path to save the dataset.
    """

    base_url: str = "https://www.cas.kth.se/COLD/db/"
    labs: list[str] = ["cold-freiburg", "cold-saarbruecken"]  # Removed ljubljana
    parts: list[str] = ["part_a", "part_b"]
    weather_conditions: list[str] = [
        "cloudy1",
        "cloudy2",
        "cloudy3",
        "cloudy4",
        "cloudy5",
        "sunny1",
        "sunny2",
        "sunny3",
        "sunny4",
        "night1",
        "night2",
        "night3",
    ]

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
                        print(
                            f"Failed to download {save_name}. HTTP status code: {response.status_code}"
                        )
                        continue

                    print(f"Extracting {save_name}...")
                    with tarfile.open(target_tar_path, "r:") as tar:
                        tar.extractall(path=extracted_seq_path)

                    extracted_contents = os.listdir(extracted_seq_path)
                    if len(extracted_contents) == 1 and os.path.isdir(
                        os.path.join(extracted_seq_path, extracted_contents[0])
                    ):
                        nested_folder = os.path.join(
                            extracted_seq_path, extracted_contents[0]
                        )
                        print(
                            f"Moving contents from nested folder {nested_folder} to {extracted_seq_path}..."
                        )

                        for item in os.listdir(nested_folder):
                            shutil.move(
                                os.path.join(nested_folder, item), extracted_seq_path
                            )

                        os.rmdir(nested_folder)

                    if os.path.exists(extracted_seq_path):
                        for subfolder in os.listdir(extracted_seq_path):
                            subfolder_path = os.path.join(extracted_seq_path, subfolder)
                            if (
                                os.path.isdir(subfolder_path)
                                and subfolder not in keep_folders
                            ):
                                print(f"Removing {subfolder_path}...")
                                shutil.rmtree(subfolder_path)
                    else:
                        print(
                            f"Expected sequence folder not found at: {extracted_seq_path}"
                        )

                    os.remove(target_tar_path)

    print("Dataset downloaded, extracted and filtered successfully.")
    return None


def prepare_data(seq_data_path: str, final_data_path: str) -> None:
    """
    Copies images to a new folder while preserving their original names and appending class labels.
    Only uses Freiburg and Saarbrucken (removed Ljubljana).

    Args:
        seq_data_path: path of the sequences data.
        final_data_path: path where images will be stored.
    """
    sequences: list[str] = os.listdir(seq_data_path)

    # Only test sequences from Freiburg and Saarbrucken
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
        "ST",
    ]

    places_file_name: str = "localization/places.lst"
    camera_folder: str = "std_cam"

    for sequence in sequences:
        if sequence.startswith("."):
            continue

        # Skip Ljubljana sequences
        if "ljubljana" in sequence.lower():
            print(f"Skipping Ljubljana sequence: {sequence}")
            continue

        sequence_path: str = os.path.join(seq_data_path, sequence)
        places_path: str = os.path.join(sequence_path, places_file_name)
        pictures_path: str = os.path.join(sequence_path, camera_folder)

        if not os.path.exists(places_path) or not os.path.exists(pictures_path):
            print(f"Skipping {sequence} - missing required folders")
            continue

        picture_to_class: dict[str, str] = {}
        with open(places_path) as file:
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

    print("\nData preparation complete!")
    print(f"Train images: {len(os.listdir(os.path.join(final_data_path, 'train')))}")
    print(f"Test images: {len(os.listdir(os.path.join(final_data_path, 'test')))}")

    return None


if __name__ == "__main__":
    path: str = "seq_data"
    data_path: str = "data"

    download_cold_data(path)
    prepare_data(path, data_path)
