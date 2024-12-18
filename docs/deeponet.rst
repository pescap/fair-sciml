DeepONet Model Documentation
============================

Overview
--------

The DeepONet model is a neural network architecture specifically designed for learning operators, making it ideal for applications involving Partial Differential Equations (PDEs). Unlike traditional neural networks, which learn mappings between finite-dimensional inputs and outputs, DeepONet learns mappings between functions (i.e., infinite-dimensional spaces). This enables it to approximate solutions of PDEs with high efficiency and flexibility.

The model operates using a two-branch architecture:

- **Branch Network**: Encodes the input functions (e.g., source terms, coefficients).
- **Trunk Network**: Encodes spatial coordinates of the solution domain.

DeepONet seamlessly combines these two components to predict scalar values corresponding to the solution fields.

Usage Example
-------------

Here's an example of how to set up and train the DeepONet model:

.. code-block:: python

    from ml.deeponet_trainer import DeepONetTrainer
    from data_loaders.local_loader import LocalLoader

    # Initialize the data loader
    file_path = "simulations/poisson_equation.h5"
    data_loader = LocalLoader(file_path)

    # Initialize the trainer
    trainer = DeepONetTrainer(
        branch_hidden_layers=[128, 128, 128],
        trunk_hidden_layers=[128, 128, 128],
        data_loader=data_loader
    )

    # Train the model
    trainer.train(epochs=10000, batch_size=32, learning_rate=0.001, metrics_file="metrics.csv")

Model Architecture
------------------

**General Design**

The DeepONet model uses a dual-branch architecture:

- **Branch Network**:
  - Dynamically adjusts the number of input neurons based on the dimensions of the input data (e.g., number of fields in the PDE).
  - Example for Poisson Equation: 1089 neurons (corresponding to grid points for `field_input_f`).
- **Trunk Network**:
  - Dynamically adjusts the input size based on the spatial dimensions (e.g., 2 for \((x, y)\)).

**Layers**

- **Branch Network**: [Dynamic Input Size, 128, 128, 128]
- **Trunk Network**: [Dynamic Input Size, 128, 128, 128]
- **Output Layer**: Single scalar output per grid point.

**Activation Function**

- **Branch Network**: ReLU
- **Trunk Network**: ReLU

**Optimizer**

- Adam optimizer with learning rate tuning.

**Metrics**

The following metrics are calculated during training and evaluation:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **L2 Relative Error**: Provides a normalized error measure.

Key Features
------------

- **Dynamic Input Handling**: 
  - The first layer of both the branch and trunk networks dynamically adjusts to the dimensions of the input data. This ensures flexibility across different PDEs and grid resolutions.
  
- **Support for Multiple Field Inputs**:
  - The branch network can encode multiple input fields, such as source terms or coefficients.

- **Efficient Data Preprocessing**:
  - Includes a Loader for local HDF5 file.

Example Training Workflow
-------------------------

1. **Load Data**: 
   - Use `LocalLoader` or a custom data loader to load simulation data in HDF5 format.
   - Preprocess data to extract branch inputs, trunk inputs, and solution values.

2. **Model Configuration**:
   - Specify the hidden layers for both the branch and trunk networks.
   - Example: `[128, 128, 128]` for both networks.

3. **Training**:
   - Train the model using the `DeepONetTrainer` class.
   - Specify hyperparameters such as batch size, learning rate, and number of epochs.

4. **Evaluation**:
   - Evaluate the trained model using metrics like MSE and L2 relative error.

Dependencies
------------

Ensure the following dependencies are installed:

- **DeepXDE**: For implementing the DeepONet model.
- **H5py**: For reading HDF5 files.
- **Scikit-learn**: For data preprocessing and splitting.
- **NumPy**: For efficient numerical operations.

Install dependencies using the following command:

.. code-block:: bash

    pip install -r ../../requirements.txt

Advanced Usage
--------------

**Configurable Branch and Trunk Layers**

The number of hidden layers and neurons per layer can be customized for specific applications by modifying the `branch_hidden_layers` and `trunk_hidden_layers` parameters in the `DeepONetTrainer` class.

**Custom Data Loader**

Users can implement their own data loaders by extending the `DataLoader` abstract base class. This allows seamless integration of new datasets and file formats.

**Extending Metrics**

To add custom evaluation metrics, modify the `evaluate` method in the `DeepONetTrainer` class.

Contributing
------------

Contributions are welcome! If you'd like to add new features, improve documentation, or fix bugs, please submit a pull request.

Contact
-------

For questions or feedback, please contact the project maintainers:

- GitHub Issues: https://github.com/pescap/fair-sciml/issues
