import torch
import torch.nn as nn


class VisualEncoderDINO(nn.Module):
    """
    Visual encoder using DINOv2 backbone for compact place recognition features.
    DINOv2 provides superior semantic understanding and spatial relationships.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        dino_model: str = "dinov2_vits14",  # or "dinov2_vitb14", "dinov2_vitl14"
    ) -> None:
        """
        Constructor of the VisualEncoderDINO class.

        Args:
            embedding_dim: dimension of the output embedding (32-128 recommended).
            dropout: dropout probability for regularization.
            dino_model: which DINOv2 model variant to use:
                - dinov2_vits14: small, 384 dim, fastest
                - dinov2_vitb14: base, 768 dim, good balance
                - dinov2_vitl14: large, 1024 dim, most accurate but slower
        """
        super().__init__()

        # Load DINOv2 model from torch hub
        self.encoder = torch.hub.load("facebookresearch/dinov2", dino_model)

        # Freeze encoder initially (can unfreeze later for fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get backbone output dimension
        if "vits14" in dino_model:
            backbone_dim = 384
        elif "vitb14" in dino_model:
            backbone_dim = 768
        elif "vitl14" in dino_model:
            backbone_dim = 1024
        elif "vitg14" in dino_model:
            backbone_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv2 model: {dino_model}")

        # Projection head for metric learning
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.BatchNorm1d(backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(backbone_dim // 2, embedding_dim),
        )

        self.embedding_dim: int = embedding_dim
        self.backbone_dim: int = backbone_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and projection head.

        Args:
            x: input images [batch_size, 3, H, W].

        Returns:
            normalized embeddings [batch_size, embedding_dim].
        """
        if x.dim() == 3:  # [C, H, W]
            x = x.unsqueeze(0)

        # Extract DINOv2 features (CLS token by default)
        features = self.encoder(x)  # [batch_size, backbone_dim]

        # Project to embedding space
        embeddings = self.projection_head(features)

        # L2 normalize for cosine similarity
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embeddings for inference.

        Args:
            x: input images [batch_size, 3, H, W].

        Returns:
            normalized embeddings [batch_size, embedding_dim].
        """
        with torch.no_grad():
            return self.forward(x)

    def unfreeze_encoder(self, num_layers: int = -1) -> None:
        """
        Unfreeze the encoder for fine-tuning.

        Args:
            num_layers: number of layers to unfreeze from the end.
                       -1 means unfreeze all layers.
        """
        if num_layers == -1:
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            encoder_params = list(self.encoder.parameters())
            for param in encoder_params[-num_layers:]:
                param.requires_grad = True


class TripletLoss(nn.Module):
    """
    Triplet loss with online hard negative mining.
    """

    def __init__(self, margin: float = 0.3) -> None:
        """
        Args:
            margin: margin for triplet loss.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: anchor embeddings [batch_size, embedding_dim].
            positive: positive embeddings [batch_size, embedding_dim].
            negative: negative embeddings [batch_size, embedding_dim].

        Returns:
            scalar loss value.
        """
        # Compute distances (euclidean distance for normalized
        # embeddings = cosine distance)

        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        # Triplet loss with margin
        loss = torch.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Args:
            temperature: temperature parameter for scaling.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            anchor: anchor embeddings [batch_size, embedding_dim].
            positive: positive embeddings [batch_size, embedding_dim].
            negatives: negative embeddings [batch_size, num_negatives, embedding_dim].

        Returns:
            scalar loss value.
        """
        batch_size = anchor.shape[0]

        # Compute similarity between anchor and positive
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [batch_size]

        # Compute similarity between anchor and all negatives
        neg_sim = (
            torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)
            / self.temperature
        )  # [batch_size, num_negatives]

        # Concatenate positive and negative similarities
        logits = torch.cat(
            [pos_sim.unsqueeze(1), neg_sim], dim=1
        )  # [batch_size, 1 + num_negatives]

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # Cross entropy loss
        loss = nn.functional.cross_entropy(logits, labels)

        return loss
