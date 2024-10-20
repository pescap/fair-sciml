import deepxde as dde
import numpy as np
import h5py
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

class DataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and return branch inputs, trunk inputs, and outputs."""
        pass

class HuggingFaceLoader(DataLoader):
    """Load data from HuggingFace repository."""
    
    def __init__(self, repo_id: str, file_name: str):
        self.repo_id = repo_id
        self.file_name = file_name
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_path = hf_hub_download(repo_id=self.repo_id, filename=self.file_name)
        return self._load_from_h5(file_path)
    
    def _load_from_h5(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        branch_inputs = []
        trunk_inputs = []
        outputs = []
        
        with h5py.File(file_path, "r") as h5file:
            for equation_name in h5file.keys():
                eq_group = h5file[equation_name]
                for session_name in eq_group.keys():
                    session_group = eq_group[session_name]
                    for sim_name in session_group.keys():
                        sim_group = session_group[sim_name]
                        
                        # Get parameters
                        source_strength = float(sim_group.attrs['parameter_source_strength'])
                        neumann_coefficient = float(sim_group.attrs['parameter_neumann_coefficient'])
                        
                        # Get data
                        coordinates = sim_group["coordinates"][:]
                        values = sim_group["values"][:]
                        
                        branch_inputs.append([source_strength, neumann_coefficient])
                        trunk_inputs.append(coordinates)
                        outputs.append(values)
        
        return (np.array(branch_inputs), 
                np.array(trunk_inputs), 
                np.array(outputs))

class LocalLoader(DataLoader):
    """Load data from local HDF5 file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return HuggingFaceLoader("", "")._load_from_h5(self.file_path)

class DeepONetTrainer:
    """Handles training of DeepONet models."""
    
    def __init__(self, 
                 branch_layers: List[int],
                 trunk_layers: List[int],
                 data_loader: DataLoader):
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.data_loader = data_loader
        self.scalers = {
            'branch': StandardScaler(),
            'trunk': StandardScaler(),
            'output': StandardScaler()
        }
        self.model = None
    
    def prepare_data(self) -> Tuple:
        """Load and preprocess data for training."""
        branch_inputs, trunk_inputs, outputs = self.data_loader.load_data()
        
        # Scale the data
        branch_inputs_scaled = self.scalers['branch'].fit_transform(branch_inputs)
        trunk_inputs_scaled = self.scalers['trunk'].fit_transform(trunk_inputs[0])
        
        # Scale outputs
        outputs_flat = outputs.flatten().reshape(-1, 1)
        outputs_scaled = self.scalers['output'].fit_transform(outputs_flat).reshape(outputs.shape)
        
        # Split the data
        branch_train, branch_test, output_train, output_test = train_test_split(
            branch_inputs_scaled, outputs_scaled, train_size=0.8, random_state=42
        )
        
        # Trunk inputs are constant
        trunk_train = trunk_inputs_scaled
        trunk_test = trunk_inputs_scaled
        
        return (branch_train, trunk_train), output_train, (branch_test, trunk_test), output_test
    
    def train(self, epochs: int = 10000, batch_size: int = 32, learning_rate: float = 0.0001) -> Tuple[dde.model.LossHistory, dict]:
        """Train the DeepONet model."""
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data()
        
        # Create dataset
        data = dde.data.TripleCartesianProd(X_train, y_train, X_test, y_test)
        
        # Create model
        net = dde.maps.DeepONetCartesianProd(
            self.branch_layers,
            self.trunk_layers,
            activation="relu",
            kernel_initializer="Glorot normal",
            num_outputs=1
        )
        
        # Compile and train
        self.model = dde.Model(data, net)
        self.model.compile("adam", lr=learning_rate, 
                     metrics=["mean squared error", "l2 relative error"])
        
        losshistory, train_state = self.model.train(epochs=epochs, batch_size=batch_size)
        print(f'Loss History type: {losshistory}')
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)
        
        return losshistory, metrics
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate model performance."""
        # Scaled metrics
        mse = dde.metrics.mean_squared_error(y_true, y_pred)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        
        # Original scale metrics
        y_true_original = self.scalers['output'].inverse_transform(y_true)
        y_pred_original = self.scalers['output'].inverse_transform(y_pred)
        mse_original = dde.metrics.mean_squared_error(y_true_original, y_pred_original)
        l2_error_original = dde.metrics.l2_relative_error(y_true_original, y_pred_original)
        
        return {
            'mse_scaled': mse,
            'l2_error_scaled': l2_error,
            'mse_original': mse_original,
            'l2_error_original': l2_error_original
        }

    def predict(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Scale inputs
        branch_scaled = self.scalers['branch'].transform(branch_input)
        trunk_scaled = self.scalers['trunk'].transform(trunk_input)
        
        # Make prediction
        prediction_scaled = self.model.predict((branch_scaled, trunk_scaled))
        
        # Inverse transform the prediction
        prediction = self.scalers['output'].inverse_transform(prediction_scaled)
        
        return prediction

def main():
    # Example usage
    repo_id = "aledhf/pde_sims"
    file_name = "simulations.h5"
    
    # Create data loader
    loader = HuggingFaceLoader(repo_id, file_name)
    
    # Create trainer
    trainer = DeepONetTrainer(
        branch_layers=[2, 128, 128, 128],
        trunk_layers=[2, 128, 128, 128],
        data_loader=loader
    )
    
    # Train model
    losshistory, metrics = trainer.train(epochs=10000, batch_size=32)
    
    # Print results
    print("Training completed. Final metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

if __name__ == "__main__":
    main()
