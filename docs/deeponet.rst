DeepONet Model Documentation
============================

Overview
--------

The DeepONet model is designed for learning operators from data, which makes it ideal for
applications involving PDEs. This section provides an example of how to use the model for training.

Usage Example
-------------

::

    from ml.deeponet_trainer import DeepONetTrainer
    from data_loaders.huggingface_loader import HuggingFaceLoader

    # Initialize the data loader
    repo_id = "aledhf/pde_sims"
    file_name = "simulations.h5"
    data_loader = HuggingFaceLoader(repo_id, file_name)

    # Initialize the trainer
    trainer = DeepONetTrainer(
        branch_layers=[2, 128, 128, 128],
        trunk_layers=[2, 128, 128, 128],
        data_loader=data_loader
    )

    # Train the model
    losshistory, metrics = trainer.train(epochs=10000, batch_size=32, learning_rate=0.0001)

Model Architecture
------------------

- **Branch Layers**: [2, 128, 128, 128]
- **Trunk Layers**: [2, 128, 128, 128]
- **Activation Function**: ReLU
- **Optimizer**: Adam

Metrics
-------

- **Mean Squared Error (MSE)**
- **L2 Relative Error**
