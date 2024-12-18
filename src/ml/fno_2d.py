import torch
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO2d
import h5py
import time
import numpy as np


class FNOTrainer:
    def __init__(self, file_path, batch_size=32, lr=1e-3, epochs=30):
        self.file_path = file_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.stats = {}
        self.model = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        with h5py.File(self.file_path, "r") as h5file:
            field_inputs = []
            solutions = []
            for session in h5file.keys():
                session_group = h5file[session]
                for simulation in session_group.keys():
                    sim_group = session_group[simulation]
                    field_data = [
                        sim_group[key][:]
                        for key in sim_group.keys()
                        if key.startswith("field_")
                    ]
                    solution = sim_group["values"][:]
                    field_inputs.append(
                        np.stack(field_data, axis=0)
                    )  # Stack fields along the channel dimension
                    solutions.append(solution)
        field_inputs = np.array(field_inputs)
        solutions = np.array(solutions)
        return field_inputs, solutions

    def normalize_data(self, field_inputs, solutions):
        """Normalize data."""
        f_mean, f_std = field_inputs.mean(), field_inputs.std()
        s_mean, s_std = solutions.mean(), solutions.std()

        field_inputs_normalized = (field_inputs - f_mean) / f_std
        solutions_normalized = (solutions - s_mean) / s_std

        self.stats = {
            "field_mean": f_mean,
            "field_std": f_std,
            "solutions_mean": s_mean,
            "solutions_std": s_std,
        }
        return field_inputs_normalized, solutions_normalized

    def prepare_data(self):
        """Prepare data for training."""
        field_inputs, solutions = self.load_data()

        # Assuming square grid
        grid_size = int(np.sqrt(field_inputs.shape[-1]))
        field_inputs = field_inputs.reshape(
            field_inputs.shape[0], -1, grid_size, grid_size
        )
        solutions = solutions.reshape(solutions.shape[0], 1, grid_size, grid_size)

        # Normalize
        field_inputs, solutions = self.normalize_data(field_inputs, solutions)

        # Convert to tensors
        field_inputs_tensor = torch.tensor(field_inputs, dtype=torch.float32)
        solutions_tensor = torch.tensor(solutions, dtype=torch.float32)

        train_size = int(0.8 * len(field_inputs_tensor))
        train_fields = field_inputs_tensor[:train_size]
        train_solutions = solutions_tensor[:train_size]
        test_fields = field_inputs_tensor[train_size:]
        test_solutions = solutions_tensor[train_size:]

        self.train_loader = DataLoader(
            TensorDataset(train_fields, train_solutions),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(test_fields, test_solutions),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return grid_size

    def initialize_model(self, grid_size, num_fields):
        """Initialize the FNO model."""
        self.model = FNO2d(
            n_modes_height=16,
            n_modes_width=16,
            hidden_channels=64,
            in_channels=num_fields,
            out_channels=1,
        )

    def train_model(self):
        """Train the model."""
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_losses = []
        test_losses = []
        l2_errors = []

        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for x, y in self.train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            # Evaluation on test set
            self.model.eval()
            test_loss = 0
            l2_error = 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    output = self.model(x)
                    test_loss += criterion(output, y).item()
                    l2_error += torch.norm(output - y) / torch.norm(y)

            test_loss /= len(self.test_loader)
            l2_error /= len(self.test_loader)

            test_losses.append(test_loss)
            l2_errors.append(l2_error.item())

            elapsed_time = time.time() - start_time

            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, "
                f"Test Loss: {test_loss:.6f}, "
                f"L2 Error: {l2_error:.6f}, Time Elapsed: {elapsed_time:.2f} seconds"
            )

        # Store the metrics for visualization
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.l2_errors = l2_errors

    def run(self):
        """Run the entire training pipeline."""
        grid_size = self.prepare_data()
        num_fields = self.train_loader.dataset[0][0].shape[
            0
        ]  # Get number of fields dynamically
        self.initialize_model(grid_size, num_fields)  # Initialize the FNO model
        self.train_model()


# Main function
def main():
    path = "simulations/poisson_equation.h5"

    trainer = FNOTrainer(file_path=path, batch_size=32, lr=1e-3, epochs=30)
    trainer.run()


if __name__ == "__main__":
    main()
