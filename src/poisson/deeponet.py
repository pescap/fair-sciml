import deepxde as dde
import numpy as np
import tensorflow as tf
import h5py
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data_from_huggingface(repo_id, file_name):
    # Download the .h5 file from Hugging Face Hub
    downloaded_file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
    
    return downloaded_file_path

def load_data_for_deeponet(h5file_path):

    branch_inputs = []
    trunk_inputs = []
    outputs = []

    with h5py.File(h5file_path, "r") as h5file:
        # Navigate to the root and look for session groups
        for equation_name in h5file.keys():
            eq_group = h5file[equation_name]
            
            # Iterate through the session groups
            for session_name in eq_group.keys():
                session_group = eq_group[session_name]
                
                # Iterate through the simulation groups within the session
                for sim_name in session_group.keys():
                    sim_group = session_group[sim_name]
                    
                    # Get the branch inputs (source strength and Neumann coefficient)
                    source_strength = float(sim_group.attrs['parameter_source_strength'])
                    neumann_coefficient = float(sim_group.attrs['parameter_neumann_coefficient'])
                    
                    # Get the trunk inputs (coordinates)
                    coordinates = sim_group["coordinates"][:]
                    
                    # Get the outputs (values at each coordinate)
                    values = sim_group["values"][:]
                    
                    branch_inputs.append([source_strength, neumann_coefficient])
                    trunk_inputs.append(coordinates)
                    outputs.append(values)

    branch_inputs = np.array(branch_inputs)
    trunk_inputs = np.array(trunk_inputs)
    outputs = np.array(outputs)

    return branch_inputs, trunk_inputs, outputs

if __name__ == "__main__":

    repo_id = "aledhf/pde_sims"
    file_name = "simulations.h5"

    # Load the .h5 file from Hugging Face repository
    h5file_path = load_data_from_huggingface(repo_id, file_name)
    
    # Transform the data from hdf5 format
    branch_inputs, trunk_inputs, outputs = load_data_for_deeponet(h5file_path)
    
    # Standardize the branch inputs, trunk inputs, and outputs
    branch_scaler = StandardScaler()
    trunk_scaler = StandardScaler()
    output_scaler = StandardScaler()
    branch_inputs_scaled = branch_scaler.fit_transform(branch_inputs)
    trunk_inputs_scaled = trunk_scaler.fit_transform(trunk_inputs[0])  # Assuming trunk inputs are identical across simulations

    # Flatten and scale the outputs
    outputs_flat = outputs.flatten().reshape(-1, 1)
    outputs_scaled = output_scaler.fit_transform(outputs_flat).reshape(outputs.shape)

    branch_train, branch_test, output_train, output_test = train_test_split(
        branch_inputs_scaled, outputs_scaled, train_size=0.8, random_state=42
    )

    # The trunk inputs are constant, so they don't need to be split
    trunk_train = trunk_inputs_scaled  
    trunk_test = trunk_inputs_scaled

    # Prepare the data for TripleCartesianProd
    X_train = (branch_train, trunk_train)  # Tuple of branch and trunk inputs
    y_train = output_train                 # Solution values
    X_test = (branch_test, trunk_test) 
    y_test = output_test              

    # Create the dataset and the model
    data = dde.data.TripleCartesianProd(X_train, y_train, X_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [2, 128, 128, 128],  # Branch network layers
        [2, 128, 128, 128],   # Trunk network layers
        activation="relu",                      
        kernel_initializer="Glorot normal",     
        num_outputs=1                         
    )

    model = dde.Model(data, net)
    model.compile("adam", 
                  lr=0.0001, 
                  metrics=["mean squared error", "l2 relative error"])

    losshistory, train_state = model.train(epochs=10000, batch_size=32)

    # Get predictions for the test set
    y_pred = model.predict(X_test)

    mse = dde.metrics.mean_squared_error(y_test, y_pred)
    l2_error = dde.metrics.l2_relative_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"L2 Relative Error: {l2_error}")

    # Calculate metrics on the original scale
    y_test_original = output_scaler.inverse_transform(y_test)
    y_pred_original = output_scaler.inverse_transform(y_pred)
    mse = dde.metrics.mean_squared_error(y_test_original, y_pred_original)
    l2_error = dde.metrics.l2_relative_error(y_test_original, y_pred_original)
    print(f"Mean Squared Error (Original Scale): {mse}")
    print(f"L2 Relative Error (Original Scale): {l2_error}")
