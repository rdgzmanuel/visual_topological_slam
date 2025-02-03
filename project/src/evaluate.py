# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader
import numpy as np

# own modules
from src.utils import (
    load_cold_data,
    Accuracy,
    load_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    test_data: DataLoader
    _, _, test_data = load_cold_data(DATA_PATH, batch_size=128, train=False)

    model: RecursiveScriptModule = load_model(name).to(device)

    # call test step and evaluate accuracy
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
            model: pytorch model.
            test_data: dataloader of test data.
            device: device of model.
            
        Returns:
            average accuracy.
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
