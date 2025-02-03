import torch
import torchvision


class CNNExtractor(torch.nn.Module):
    """
    This class is to implement a CNN that serves as a feature extractor.
    """

    # define attributes
    transformer: torch.nn.Module
    classifier: torch.nn.Module

    def __init__(self, output_size: int = 10, dropout=0.5):
        """
        Constructor for CNNExtractor class.
        Args:
            output_size: output size for the model.
            dropout: dropout rate for regularization.
        """
        super().__init__()

        self.transformer = torchvision.models.resnet101(pretrained=True)

        num_features: int = 2048
        intermediate_size: int = num_features // 2
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(num_features, intermediate_size),
            torch.nn.ReLU(),
            torch.nn.Linear(intermediate_size, output_size),
        )

        for param in self.transformer.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: batch of images. Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, number of classes].
        """
        features = self.transformer(inputs)
        
        features = features.view(features.size(0), -1)
        return self.classifier(features)
