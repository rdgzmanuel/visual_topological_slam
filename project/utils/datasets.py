import os
import torch
import tarfile
import requests
import shutil
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


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

        # TODO
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        labels_correspondence: dict[str: int]
        class_number: int = 0

        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        for image in os.listdir(path):
            image_splitted: list[str] = image[:-5].split("_")  # Remove the .jpeg
            label_name: str = image_splitted[-1]
            if label_name not in labels_correspondence:
                labels_correspondence[label_name] = class_number
                class_number += 1

            image_path: str = os.path.join(path, image)
            open_image = Image.open(image_path)
            tensor_image: torch.Tensor = transform(open_image)
            self.images.append(tensor_image)
            self.labels.append(labels_correspondence[label_name])
        

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        # TODO
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """

        # TODO
        image: torch.Tensor = self.images[index]
        label: int = self.labels[index]

        return image, label


def load_cold_data(
    path: str, batch_size: int = 128, num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns three Dataloaders, one for train data, one for val data and
    other for testing data for COLD dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_cold_data(path)

    # create datasets
    train_dataset: Dataset = COLDataset(f"{path}/train")
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
    test_dataset: Dataset = COLDataset(f"{path}/test")

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


def download_cold_data(path: str) -> None:
    """
    Downloads and extracts the COLD dataset, keeping only specific subfolders inside each sequence folder.

    Args:
        path: Path to save the dataset.
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
                    picture_to_class[parts[0]] = parts[1]

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


if __name__ == "__main__":

    path: str = "seq_data"
    data_path: str = "data"
    # download_cold_data(path)
    # prepare_data(path, data_path)
    load_cold_data(data_path)