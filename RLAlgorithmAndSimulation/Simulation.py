import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the model and scalers
def load_models_and_scalers():
    forest_model_outputs = joblib.load('random_forest_modelPC_simple.pkl')
    scaler_X = joblib.load('scaler_X_simple.pkl')
    scaler_y = joblib.load('scaler_y_simple.pkl')
    
    return {
        "forest_model_outputs": forest_model_outputs,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }

# Initialize inputs from saved dataset
def initialize_inputs():
    # Load initial input data
    file_path = "fullinputs_new.csv" 
    X_test = joblib.load('X_test_forest_modelPC_simple.pkl')
    inputs = X_test.iloc[0].values.copy()
    return inputs
    
# Convert action into controllable inputs for simulation
def actions_to_control_inputs(inputs, action):
    # Action consists of: [fw_pump_speed, sw_pump_speed, valve_sw_position]
    fw_pump_speed = action[0]
    if fw_pump_speed <= 100:
        inputs[0] = fw_pump_speed
        inputs[1] = 0
        inputs[2] = 0
    elif 100 < fw_pump_speed <= 200:
        inputs[0] = 100
        inputs[1] = fw_pump_speed - 100
        inputs[2] = 0
    else:
        inputs[0] = 100
        inputs[1] = 100
        inputs[2] = fw_pump_speed - 200

    sw_pump_speed = action[1]
    if sw_pump_speed <= 100:
        inputs[3] = sw_pump_speed
        inputs[4] = 0
    else:
        inputs[3] = 100
        inputs[4] = sw_pump_speed - 100

    # Set valve control
    inputs[5] = action[2]


    return inputs

# Simulation step: Given inputs and action, returns predicted outputs
def simulate_step(inputs, model, scaler_X, scaler_y):
    # Scale inputs before passing to model
    inputs_scaled = scaler_X.transform([inputs]) 
    prediction = model.predict(inputs_scaled)  # Make prediction
    prediction_unscaled = scaler_y.inverse_transform(prediction)  # Inverse transform to get actual values
    return prediction_unscaled.flatten()

def set_Tsw_PC(inputs, previous_Tsw_PC=None):
    if previous_Tsw_PC is None:
        # If it's the first run, set a random value
        Tsw_PC_value = np.random.uniform(2.3, 9)
    else:
        # Generate a random value close to the previous one
        Tsw_PC_value = np.clip(np.random.uniform(previous_Tsw_PC - 0.01, previous_Tsw_PC + 0.01), 2.3, 9)

    
    inputs[6] = Tsw_PC_value  # Assuming index 6 corresponds to Tsw_PC
    return inputs, Tsw_PC_value


def save_data(q_table, reward_over_time, predictions_over_time, control_inputs_over_time, 
              full_inputs_over_time, elapsed_time_sample, output_names, controllable_input_names, full_input_names):
    """
    Save Q-table, reward, predictions, control inputs, and other data to CSV and pickle files.
    """
    # Save Q-table
    q_table.to_csv("q_table_new.csv")
    q_table.to_pickle("q_table_new.pkl")
    print("Q-table saved as q_table_new.csv and q_table_new.pkl")

    # Save reward over time
    pd.DataFrame(reward_over_time, columns=['reward']).to_csv('reward_over_time_new.csv', index=False)
    print("Reward data saved to reward_over_time_new.csv")

    # Save predictions
    pd.DataFrame(predictions_over_time, columns=output_names).to_csv('predictions_new.csv', index=False)
    print("Predictions data saved to predictions_new.csv")

    # Save control inputs
    pd.DataFrame(control_inputs_over_time, columns=controllable_input_names).to_csv('control_inputs_over_time_new.csv', index=False)
    print("Control inputs data saved to control_inputs_over_time_new.csv")

    # Save full inputs
    pd.DataFrame(full_inputs_over_time, columns=full_input_names).to_csv('fullinputs_new.csv', index=False)
    print("Full inputs data saved to fullinputs_new.csv")

    # Save elapsed time
    pd.DataFrame(elapsed_time_sample, columns=['time']).to_csv('elapsed_time_new.csv', index=False)
    print("Elapsed time data saved to elapsed_time_new.csv")


def plot_data(reward_over_time, predictions_over_time, control_inputs_over_time, output_names, 
              controllable_input_names, deltaP_sp=None, Tfw_sp=None):
    """
    Plot reward, predictions, controllable inputs, and Q-table heatmap.
    """
    # Plot reward over time
    plt.figure(figsize=(10, 6))
    plt.plot(reward_over_time, label="Reward")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_over_time_new.png')
    plt.show()

    # Plot predicted outputs over time
    predictions = np.array(predictions_over_time)
    fig, axes = plt.subplots(len(output_names), 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Predicted Outputs Over Time', fontsize=16)

    for i, ax in enumerate(axes):
        ax.plot(predictions[:, i], label=output_names[i])
        if i == 1 and deltaP_sp is not None:
            ax.axhline(deltaP_sp, color='r', linestyle='--', label='deltaP_sp')
        if i == 3 and Tfw_sp is not None:
            ax.axhline(Tfw_sp, color='g', linestyle='--', label='Tfw_sp')
        ax.set_ylabel(output_names[i])
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('OutputRL.png')
    plt.show()

    # Plot controllable inputs over time
    control_inputs = np.array(control_inputs_over_time)
    fig, axes = plt.subplots(len(controllable_input_names), 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Controllable Inputs Over Time', fontsize=16)

    for i, ax in enumerate(axes):
        ax.plot(control_inputs[:, i], label=controllable_input_names[i])
        ax.set_ylabel(controllable_input_names[i])
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('InputRL.png')
    plt.show()
