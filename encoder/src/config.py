from pathlib import Path

SEQ_DATA_PATH: str = "seq_data"
DATA_PATH: str = "data"
IMAGES_PATH: Path = Path("images")

HYPERPARAMETERS: dict = {
    "contrastive": {
        "epochs": 15,
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "batch_size": 32,
        "embedding_dim": 128,
        "dropout": 0.0,
        "temperature": 0.07,
        "weight_decay": 1e-5,
        "eta_min": 1e-6,
    },
    "triplet": {
        "epochs": 30,
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "batch_size": 32,
        "embedding_dim": 128,
        "dropout": 0.0,
        "margin": 0.5,
        "weight_decay": 1e-5,
        "step_size": 15,
        "gamma": 0.5,
    },
}
