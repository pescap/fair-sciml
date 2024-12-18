FNO 2D Model Documentation
===========================

Overview
--------

The Fourier Neural Operator (FNO) is a neural network architecture designed for solving partial differential equations (PDEs) by learning operators in the Fourier domain. The FNO leverages Fourier transforms to efficiently model spatial relationships across different resolutions, making it ideal for applications with complex dynamics and high-resolution data.

This implementation focuses on the 2D case and is integrated into the `fair-sciml` project for learning solutions of PDEs from simulation data.

Usage Example
-------------

Below is an example of how to use the implemented FNO 2D model for training:

.. code-block:: python

    from ml.fno_trainer import FNOTrainer

    # Initialize the trainer
    trainer = FNOTrainer(
        file_path="simulations/poisson_equation.h5",
        batch_size=32,
        lr=1e-3,
        epochs=50
    )

    # Run the training pipeline
    trainer.run()

Model Architecture
------------------

The FNO 2D model operates on grid-based data with the following architecture:

- **Input Channels**:
  - Number of input channels corresponds to the number of input fields (e.g., `field_input_f`).
  - Example for Poisson Equation: 2 channels (source field, boundary field).
- **Hidden Channels**:
  - Default: 64 hidden channels in the intermediate Fourier layers.
- **Output Channels**:
  - 1 output channel, representing the solution field.
  
**Layers**

- **Fourier Layers**:
  - Captures global spatial dependencies using Fourier transforms.
  - Operates on the input grid with specified modes in the height and width dimensions.
- **Modes**:
  - `n_modes_height=16`
  - `n_modes_width=16`
- **Final Projection Layer**:
  - Maps the output of Fourier layers to the solution domain.

**Activation Function**

- ReLU activation is applied in intermediate layers.

**Optimizer**

- Adam optimizer with configurable learning rate.

Key Features
------------

- **Discretization Invariance**:
  - The FNO 2D model can adapt to various grid resolutions without retraining.
- **Fourier Transform-Based Layers**:
  - Efficiently captures both global and local spatial dynamics.
- **Dynamic Input Handling**:
  - Supports flexible input sizes and multiple input fields.
- **Efficient Data Handling**:
  - Automatically normalizes and prepares data for training.

Training Workflow
-----------------

1. **Load Data**:
   - HDF5 files are loaded using the `load_data` method.
   - Input fields and solution fields are extracted and normalized.
   - Example fields: `field_input_f`, solution values.

2. **Prepare Data**:
   - Data is reshaped into tensors for training and testing.
   - Grid size is calculated dynamically based on the input data.

3. **Model Initialization**:
   - The number of input channels is set dynamically based on the number of fields in the HDF5 file.

4. **Training**:
   - The model is trained using MSE loss and the Adam optimizer.
   - Metrics such as test loss and L2 relative error are logged during training.

Metrics
-------

The following metrics are logged during training and evaluation:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predictions and ground truth.
- **L2 Relative Error**: Normalized error that accounts for the magnitude of the true solution.

Dependencies
------------

Ensure the following dependencies are installed:

- **NeuralOperator**: For implementing the FNO model.
- **PyTorch**: For deep learning operations.
- **H5py**: For reading HDF5 files.
- **NumPy**: For efficient numerical operations.

Install dependencies using the following command:

.. code-block:: bash

    pip install -r ../../requirements.txt

Advanced Usage
--------------

**Configurable Fourier Modes**

You can adjust the number of Fourier modes in the height and width dimensions by modifying the `n_modes_height` and `n_modes_width` parameters in the `FNOTrainer` class.

**Multi-Field Inputs**

The FNO model supports multiple input fields, such as source terms and coefficients. Simply add these fields to the HDF5 file and ensure the data loader correctly extracts them.

**Extended Metrics**

To include custom evaluation metrics, modify the `train_model` method in the `FNOTrainer` class.

Contact
-------

For questions or feedback, please contact the project maintainers:

- GitHub Issues: https://github.com/pescap/fair-sciml/issues
