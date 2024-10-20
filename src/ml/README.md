# **Machine Learning Models in fair-sciml**

This directory contains the **machine learning components** of the `fair-sciml` project. Currently, it includes the implementation of the **DeepONet model**, which is designed to learn operators from simulation data generated by PDE solvers.

---

## **Contents**

- [Overview](#overview)
- [DeepONet](#deeponet)
- [Usage](#usage)
- [Training Workflow](#training-workflow)
- [Dependencies](#dependencies)
- [Contact](#contact)

---

## **Overview**

The machine learning models in this directory focus on **operator learning** from the results of PDE simulations.

---

## **DeepONet**

The **DeepONet model** is implemented to predict PDE solutions given input parameters (branch inputs) and spatial coordinates (trunk inputs). It consists of:

- **Branch Network**: Encodes input parameters.
- **Trunk Network**: Encodes spatial coordinates of the solution domain.
- **Output**: Predicts scalar values corresponding to PDE solutions.

---

## **Usage**

You can train the DeepONet model using the `deeponet_trainer.py` script. Make sure the necessary simulation data (in HDF5 format) is available.

**Example:**

```python
python3 deeponet_trainer.py
```

---

## **Training Workflow**

1. **Load Data**: The `HuggingFaceLoader` or `LocalLoader` is used to load simulation data.
2. **Preprocess Data**: Data is scaled and split into training and testing sets.
3. **Model Creation**: A DeepONet model with specified branch and trunk networks is built.
4. **Training**: The model is trained using the Adam optimizer.
5. **Evaluation**: Metrics such as MSE and L2 relative error are calculated.

---

## **Dependencies**

- **DeepXDE**: For implementing DeepONet.  
- **H5py**: For reading HDF5 files.  
- **HuggingFace Hub**: For downloading datasets from HuggingFace.  
- **Scikit-learn**: For data preprocessing and splitting.  

Ensure all dependencies are installed by running:

---

```bash
pip install -r ../../requirements.txt
```
