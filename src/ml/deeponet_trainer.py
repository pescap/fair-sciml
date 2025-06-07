import deepxde as dde
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Abstract base class for data loading."""

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and return branch inputs, trunk inputs, and outputs."""
        pass


class LocalLoader(DataLoader):
    """Load data from local HDF5 file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        field_inputs = []
        trunk_inputs = []
        outputs = []

        with h5py.File(self.file_path, "r") as h5file:
            for session_name in h5file.keys():
                session_group = h5file[session_name]
                for sim_name in session_group.keys():
                    sim_group = session_group[sim_name]


                    # Get trunk inputs (coordinates) and outputs (values)
                    coordinates = sim_group["coordinates"][:]
                    values = sim_group["values"][:]

                    trunk_inputs.append(np.float32(coordinates))
                    outputs.append(np.float32(values))
                    field_inputs.append(np.ones(values.shape, dtype=np.float32)*np.float32(sim_group.attrs["parameter_coefficient"]))

        return (
            np.array(field_inputs),
            np.array(trunk_inputs),
            np.array(outputs),
        )


class DeepONetTrainer:
    """Handles training of DeepONet models."""

    def __init__(
        self,
        branch_hidden_layers: List[int],
        trunk_hidden_layers: List[int],
        data_loader: DataLoader,
    ):
        self.branch_hidden_layers = branch_hidden_layers
        self.trunk_hidden_layers = trunk_hidden_layers
        self.data_loader = data_loader
        self.model = None

    def prepare_data(self) -> Tuple:
        """Load and preprocess data for training."""
        field_inputs, spatial_inputs, outputs = self.data_loader.load_data()

        branch_inputs = field_inputs[..., 0]  # Shape (n_samples, n_points)

        trunk_inputs = spatial_inputs[0]  # Use spatial points from the first sample

        # Flatten the branch inputs to match expected dimensions
        branch_inputs_flattened = branch_inputs.reshape(branch_inputs.shape[0], -1)

        # Split the data
        branch_train, branch_test, output_train, output_test = train_test_split(
            branch_inputs_flattened, outputs, train_size=0.8, random_state=42
        )
        print('branch_train shape:', branch_train.shape)
        print('branch_test shape:', branch_test.shape)
        print('output_train shape:', output_train.shape)
        print('output_test shape:', output_test.shape)
        # Flatten outputs
        output_train = output_train.reshape(
            output_train.shape[0], -1
        )  # Shape (n_samples, n_points)
        output_test = output_test.reshape(
            output_test.shape[0], -1
        )  # Shape (n_samples, n_points)
        print('output_train shape after flattening:', output_train.shape)
        print('output_test shape after flattening:', output_test.shape)
        trunk_train, trunk_test = trunk_inputs, trunk_inputs
        print('trunk_train shape:', trunk_train.shape)
        print('trunk_test shape:', trunk_test.shape)
        return (
            (branch_train, trunk_train),
            output_train,
            (branch_test, trunk_test),
            output_test,
        )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        mse = np.mean((y_true - y_pred) ** 2)
        l2_relative_error = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

        return {
            "mean_squared_error": mse,
            "l2_relative_error": l2_relative_error,
        }

    def train(
        self,
        epochs: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        metrics_file: str = None,
    ) -> None:
        """Train the DeepONet model."""
        # Prepare data
        (
            (branch_train, trunk_train),
            output_train,
            (branch_test, trunk_test),
            output_test,
        ) = self.prepare_data()

        # Define DeepONet architecture
        n_fields = branch_train.shape[1]  # Number of fields
        n_dims = trunk_train.shape[1]  # Spatial dimensions (e.g., x, y)

        net = dde.maps.DeepONetCartesianProd(
            [n_fields] + self.branch_hidden_layers,
            [n_dims] + self.trunk_hidden_layers,
            activation="relu",
            kernel_initializer="Glorot normal",
            num_outputs=1,
        )

        # Create dataset
        data = dde.data.TripleCartesianProd(
            (branch_train, trunk_train),
            output_train,
            (branch_test, trunk_test),
            output_test,
        )

        # Compile and train
        self.model = dde.Model(data, net)
        self.model.compile(
            "adam",
            lr=learning_rate,
            metrics=["mean squared error", "l2 relative error"],
        )

        losshistory, train_state = self.model.train(
            epochs=epochs, batch_size=batch_size
        )


def main():
    # Example usage
    file_path = "simulations/biharmonic_equation.h5"

    # Create data loader
    loader = LocalLoader(file_path)

    # Define trainer with model configuration
    trainer = DeepONetTrainer(
        branch_hidden_layers=[128, 128, 128],
        trunk_hidden_layers=[128, 128, 128],
        data_loader=loader,
    )

    # Train model and save metrics
    trainer.train(
        epochs=20000,
        batch_size=32,
        learning_rate=1e-3,
        metrics_file="deeponet_metrics.csv",
    )


if __name__ == "__main__":
    main()
