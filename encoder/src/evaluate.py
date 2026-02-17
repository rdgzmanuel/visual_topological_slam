import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models import VisualEncoder
from src.utils import load_evaluation_data, load_model, set_seed

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

set_seed(42)
torch.set_num_threads(8)

DATA_PATH: str = "data"
SEQ_DATA_PATH: str = "seq_data"


def main(model_name: str) -> dict[str, float]:
    """
    Evaluate the visual encoder on place recognition metrics.

    Args:
        model_name: name of the saved model.

    Returns:
        dictionary with evaluation metrics.
    """

    # Load test data (should return query and database sets)
    query_data: DataLoader
    database_data: DataLoader
    query_data, database_data = load_evaluation_data(
        seq_data_path=SEQ_DATA_PATH, data_path=DATA_PATH, batch_size=64
    )

    model: VisualEncoder = load_model(model_name).to(device)

    metrics = evaluate_place_recognition(model, query_data, database_data, device)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("=" * 50 + "\n")

    return metrics


def evaluate_place_recognition(
    model: VisualEncoder,
    query_data: DataLoader,
    database_data: DataLoader,
    device: torch.device,
    top_k_values: list[int] = [1, 5, 10, 20],
) -> dict[str, float]:
    """
    Evaluate place recognition performance using recall@K metrics.

    Args:
        model: trained visual encoder.
        query_data: dataloader for query images.
        database_data: dataloader for database images.
        device: computation device.
        top_k_values: list of K values for recall@K metric.

    Returns:
        dictionary containing recall@K for each K value.
    """
    model.eval()

    # Extract embeddings for all database images
    print("Extracting database embeddings...")
    database_embeddings: list[np.ndarray] = []
    database_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(database_data, desc="Database"):
            images = images.to(device)
            embeddings = model.extract_features(images)
            database_embeddings.append(embeddings.cpu().numpy())
            database_labels.extend(labels.numpy())

    database_embeddings = np.vstack(database_embeddings)
    database_labels = np.array(database_labels)

    # Extract embeddings for all query images
    print("Extracting query embeddings...")
    query_embeddings: list[np.ndarray] = []
    query_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(query_data, desc="Queries"):
            images = images.to(device)
            embeddings = model.extract_features(images)
            query_embeddings.append(embeddings.cpu().numpy())
            query_labels.extend(labels.numpy())

    query_embeddings = np.vstack(query_embeddings)
    query_labels = np.array(query_labels)

    # Compute similarity matrix (using cosine similarity since embeddings are normalized)
    print("Computing similarities...")
    similarity_matrix = cosine_similarity(query_embeddings, database_embeddings)

    # Compute Recall@K
    metrics: dict[str, float] = {}

    for k in top_k_values:
        recall = compute_recall_at_k(
            similarity_matrix, query_labels, database_labels, k
        )
        metrics[f"recall@{k}"] = recall
        print(f"Recall@{k}: {recall:.4f}")

    # Compute mean average precision
    map_score = compute_mean_average_precision(
        similarity_matrix, query_labels, database_labels
    )
    metrics["mAP"] = map_score
    print(f"Mean Average Precision: {map_score:.4f}")

    # Compute embedding statistics
    print("\nEmbedding Statistics:")
    print(f"Query embedding shape: {query_embeddings.shape}")
    print(f"Database embedding shape: {database_embeddings.shape}")
    print(
        f"Query embedding norm (mean): {np.linalg.norm(query_embeddings, axis=1).mean():.4f}"
    )
    print(
        f"Database embedding norm (mean): {np.linalg.norm(database_embeddings, axis=1).mean():.4f}"
    )

    return metrics


def compute_recall_at_k(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    database_labels: np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K metric.

    Args:
        similarity_matrix: [num_queries, num_database] similarity scores.
        query_labels: ground truth labels for queries.
        database_labels: ground truth labels for database.
        k: number of top retrievals to consider.

    Returns:
        recall@K score.
    """
    num_queries = similarity_matrix.shape[0]
    correct: int = 0

    for i in range(num_queries):
        # Get top-k most similar database images
        top_k_indices = np.argsort(similarity_matrix[i])[-k:]
        top_k_labels = database_labels[top_k_indices]

        # Check if correct label is in top-k
        if query_labels[i] in top_k_labels:
            correct += 1

    recall = correct / num_queries
    return recall


def compute_mean_average_precision(
    similarity_matrix: np.ndarray, query_labels: np.ndarray, database_labels: np.ndarray
) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        similarity_matrix: [num_queries, num_database] similarity scores.
        query_labels: ground truth labels for queries.
        database_labels: ground truth labels for database.

    Returns:
        mean average precision score.
    """
    num_queries = similarity_matrix.shape[0]
    average_precisions: list[float] = []

    for i in range(num_queries):
        # Sort database by similarity (descending)
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        sorted_labels = database_labels[sorted_indices]

        # Compute average precision for this query
        relevant = (sorted_labels == query_labels[i]).astype(int)

        if relevant.sum() == 0:
            continue

        precisions: list[float] = []
        num_relevant: int = 0

        for j, is_relevant in enumerate(relevant):
            if is_relevant:
                num_relevant += 1
                precision = num_relevant / (j + 1)
                precisions.append(precision)

        if precisions:
            average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions) if average_precisions else 0.0


if __name__ == "__main__":
    model_name = "visual_encoder_triplet_dim128_best"
    metrics = main(model_name)
