import numpy as np
import random
import pandas as pd
from SimulationTestSimpleNew import initialize_inputs, load_models_and_scalers, actions_to_control_inputs, simulate_step, set_Tsw_PC, set_Tsw_PC, save_data, plot_data
import warnings

warnings.filterwarnings("ignore")

# Load the model
models_and_scalers = load_models_and_scalers()
models_and_scalers = load_models_and_scalers()
forest_model_outputs = models_and_scalers["forest_model_outputs"]
scaler_X = models_and_scalers["scaler_X"]
scaler_y = models_and_scalers["scaler_y"]

previous_Tsw_PC = None  # Initialize previous Tsw_PC

# Initialize inputs
inputs = initialize_inputs()

# Initiate setpoint
deltaP_sp = 0.50
Tfw_sp = 10.0

#Initialize RL paramters
alpha, gamma, epsilon, decay = 1, 0.9, 1.0, 0.9999

# Define action
vfw_actions = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
vsw_actions = [0, 25, 50, 75, 100, 125, 150, 175, 200]
valve_sw_actions = [0, 25, 50, 75, 100]

# Defining states
deltaP_14_states = [(-np.inf, -0.5), (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0),(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, np.inf)]
Tfw_states = [(-np.inf, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0),(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, np.inf)]
Tsw_PC_states = [(-np.inf, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, np.inf)]

# Initialize Q-table based on the above state ranges
def create_q_table(deltaP_14_states, Tfw_states, Tsw_PC_states, vfw_actions, vsw_actions, valve_sw_actions):
    # Generate all possible combinations of the state
    states = [(dp, fw, sw) for dp in range(len(deltaP_14_states))
              for fw in range(len(Tfw_states))
              for sw in range(len(Tsw_PC_states))]

    # Generate all possible combinations of actions
    actions = [(vfw, vsw, valve_sw) for vfw in vfw_actions
           for vsw in vsw_actions
           for valve_sw in valve_sw_actions]
    # Create a Q-table
    q_table = pd.DataFrame(
        0.0, 
        index=pd.MultiIndex.from_tuples(states, names=["deltaP_14 deviation", "Tfw deviation", "Tsw_PC state"]),
        columns=pd.MultiIndex.from_tuples(actions, names=["Vfw", "Vsw Action", "Valve_SW Action"])
    )
    return q_table

def choose_action_epsilon_greedy(q_table, current_state, epsilon):
    state_q_values = q_table.loc[current_state]
    if np.random.uniform(0, 1) < epsilon:
        action = random.choice(q_table.columns)
    else:
        # Find all actions with the maximum Q-value
        max_q_value = state_q_values.max()
        best_actions = state_q_values[state_q_values == max_q_value].index.tolist()
        action = random.choice(best_actions)  # Randomly choose among the best actions
    return action

def update_q_table(q_table, current_state, action, reward, new_state, alpha, gamma):
    current_q_value = q_table.loc[current_state, action]
    max_future_q_value = q_table.loc[new_state].max()
    new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
    q_table.loc[current_state, action] = new_q_value
    return q_table

def discretize_state(deltaP_14, Tfw, Tsw_PC, deltaP_14_states, Tfw_states, Tsw_PC_states, deltaP_sp, Tfw_sp):
    # Calculate deviations from setpoints
    deltaP_deviation = deltaP_14 - deltaP_sp
    Tfw_deviation = Tfw - Tfw_sp

    def find_state(value, states):
        for i, (low, high) in enumerate(states):
            if low <= value < high:
                return i
        return len(states) - 1  # Assign to the last interval if out of range

    deltaP_state = find_state(deltaP_deviation, deltaP_14_states)
    Tfw_state = find_state(Tfw_deviation, Tfw_states)
    Tsw_PC_state = find_state(Tsw_PC, Tsw_PC_states)
    
    return (deltaP_state, Tfw_state, Tsw_PC_state)

def compute_reward(deltaP_14, deltaP_sp, Tfw, Tfw_sp, vfw1, vfw2, vfw3, vsw1, vsw2, valve_sw):
    """
    Compute the reward based on the deviation from the setpoints.
    """
    vfw = vfw1 + vfw2 + vfw3
    vsw = vsw1 + vsw2

    if deltaP_14 > deltaP_sp:
        r_deltaP = -100 * (deltaP_14 - deltaP_sp) ** 2
    else:  # When deltaP_78 <= deltaP_sp
        r_deltaP = -100 * (deltaP_sp-deltaP_14)

    if Tfw > Tfw_sp:
        r_Tfw = -10 * abs(Tfw - Tfw_sp)
    else:  # When Tfw <= Tfw_sp
        r_Tfw = -1 * abs(Tfw_sp - Tfw) ** 2

    r_vsw = -1/8 * vsw

    r_vfw = -1/12 * vfw

    r_valve = -1/2 * valve_sw

    total_reward = r_deltaP + r_Tfw  + r_valve + r_vfw + r_vsw
    return total_reward


#Initialize paramters and simulations
initial_prediction = simulate_step(inputs, forest_model_outputs, scaler_X, models_and_scalers["scaler_y"])
prediction = initial_prediction  # Initialize predictions with the initial output
previous_predictions = np.zeros(5)
q_table = create_q_table(deltaP_14_states, Tfw_states, Tsw_PC_states, vfw_actions, vsw_actions, valve_sw_actions)
print(q_table.shape)

#Lists to store for plots
control_inputs_over_time = []
predictions_over_time = [] 
reward_over_time = []
full_inputs_over_time = []
elapsed_time_sample = []
controllable_inputs_indices = [0, 1, 2, 3, 4, 5, 6]

#Loop paramters
num_episodes = 500000  #safety cap
consecutive_iterations = 0
stop_after_iterations = 1160
# Define the time interval
time_interval = 300  # seconds
elapsed_time = 0 

for episode in range(num_episodes):

    if consecutive_iterations >= stop_after_iterations:
        print(f"\nTraining completed early after {episode} episodes: deltaP_14 = {deltaP_14}, Tfw = {Tfw} for {consecutive_iterations} consecutive iterations.")
        break

    print(f"Episode {episode + 1}/{num_episodes} Epsilon {epsilon}", end="\r")

    # Set the Tsw_PC
    inputs, previous_Tsw_PC = set_Tsw_PC(inputs, previous_Tsw_PC)

    # Find current state
    current_state = discretize_state(prediction[1], prediction[3], inputs[6], deltaP_14_states, Tfw_states, Tsw_PC_states, deltaP_sp, Tfw_sp)

    # Choose action
    action = choose_action_epsilon_greedy(q_table, current_state, epsilon)
    
    # Update inputs based on action
    inputs = actions_to_control_inputs(inputs, action)
    current_control_inputs = [inputs[idx] for idx in controllable_inputs_indices]
    control_inputs_over_time.append(current_control_inputs)

    # set inputs for simulation
    full_inputs = np.concatenate([inputs[:7], [time_interval], [elapsed_time], previous_predictions])  # Current controllable inputs + previous predictions
    full_inputs_over_time.append(full_inputs)

    # Run the simulation step with the current full input
    prediction = simulate_step(full_inputs, forest_model_outputs, scaler_X, scaler_y)
    predictions_over_time.append(prediction)
    previous_predictions = prediction  # previous prediction for the next iteration

    # Compute reward
    deltaP_14 = prediction[1]
    Tfw = prediction[3]
    vfw1 = inputs[0]
    vfw2 = inputs[1]
    vfw3 = inputs[2]
    vsw1 = inputs[3]
    vsw2 = inputs[4]
    valve_sw = inputs[5]
    Tsw_PC = inputs[6]

    new_state = discretize_state(deltaP_14, Tfw, Tsw_PC, deltaP_14_states, Tfw_states, Tsw_PC_states, deltaP_sp, Tfw_sp)

    # Check if the state has changed
    reward = compute_reward(deltaP_14, deltaP_sp, Tfw, Tfw_sp, vfw1, vfw2, vfw3, vsw1, vsw2, valve_sw)
    reward_over_time.append(reward)

    # Update Q-table
    q_table = update_q_table(q_table, current_state, action, reward, new_state, alpha, gamma)

    # Check if conditions are upheld
    if -0.05 <= (deltaP_14 - deltaP_sp) < 0.05 and -0.5 <= (Tfw_sp - Tfw) <= 1 :#and epsilon < 0.01:
        consecutive_iterations += 1
    else:
        consecutive_iterations = 0  # Reset the counter if the condition is not met

    # change elapsed time
    elapsed_time += time_interval
    elapsed_time_sample.append(elapsed_time)

    #decay paramters
    epsilon = max(epsilon * decay, 0.00)
    alpha = max(alpha * decay, 0.1)



output_names = ['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw']
controllable_input_names = ['vfw_1', 'vfw_2', 'vfw_3', 'vsw_1', 'vsw_2', 'Valve_sw','Tsw_PC']
full_input_names = ['vfw_1', 'vfw_2', 'vfw_3', 'vsw_1', 'vsw_2', 'Valve_sw','Tsw_PC','interval','elapsed_time','previous_deltaP_78', 'previous_deltaP_14', 'previous_deltaP_12', 'previous_Tfw', 'previous_ffw']

predictions = np.array(predictions_over_time)
control_inputs_over_time = np.array(control_inputs_over_time)
control_inputs_over_time = control_inputs_over_time[:, :7]

# saving and plotting data for vizualization
save_data(q_table, reward_over_time, predictions_over_time, control_inputs_over_time, 
          full_inputs_over_time, elapsed_time_sample, output_names, controllable_input_names, full_input_names)


plot_data(reward_over_time, predictions_over_time, control_inputs_over_time, 
          output_names, controllable_input_names, deltaP_sp, Tfw_sp)
