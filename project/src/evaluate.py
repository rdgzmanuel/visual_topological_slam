import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader
import numpy as np

from src.utils import (
    load_cold_data,
    Accuracy,
    load_model,
    set_seed,
)

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

set_seed(42)
torch.set_num_threads(8)

DATA_PATH: str = "data"
SEQ_DATA_PATH: str = "seq_data"


def main(name: str) -> float:
    """
    This is the main function of the program. Performs accuracy evaluation.

    Args:
        name (str): name of the model to evaluate.

    Returns:
        float: accuracy of the model.
    """

    test_data: DataLoader
    _, _, test_data = load_cold_data(seq_data_path=SEQ_DATA_PATH, data_path=DATA_PATH, batch_size=128, train=False)

    model: RecursiveScriptModule = load_model(name).to(device)

    accuracy: float = t_step(model, test_data, device)

    return accuracy


def t_step(
        model: torch.nn.Module,
        test_data: DataLoader,
        device: torch.device,
    ) -> float:
        """
        This function computes the test step.

        Args:
            model (torch.nn.Module): pytorch model.
            test_data (DataLoader): dataloader of test data.
            device (torch.device): device of model.
            
        Returns:
            float: average accuracy.
        """

        model.eval()

        accuracy: Accuracy = Accuracy()

        with torch.no_grad():
            accuracies: list[float] = []
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs: torch.Tensor = model(images)
                
                accuracy.update(outputs, labels.long())
                accuracy_value: float = accuracy.compute()

                accuracies.append(accuracy_value)
            
            final_accuracy: float = float(np.mean(accuracies))
        return final_accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
