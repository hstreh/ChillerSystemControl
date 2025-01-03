from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
from keras import models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_pathChiller = r"C:\Users\HETS\Desktop\Helena Skole\PCdata.csv"
ChillerData = pd.read_csv(file_pathChiller)

# Convert Time to datetime format
ChillerData['Time'] = pd.to_datetime(ChillerData['Time'], format='%d/%m/%Y %H:%M:%S.%f')

# Extract useful time features
ChillerData['microsecond'] = ChillerData['Time'].dt.microsecond
ChillerData['second'] = ChillerData['Time'].dt.second
ChillerData['minute'] = ChillerData['Time'].dt.minute
ChillerData['hour'] = ChillerData['Time'].dt.hour
ChillerData['day'] = ChillerData['Time'].dt.day
ChillerData['month'] = ChillerData['Time'].dt.month
ChillerData['year'] = ChillerData['Time'].dt.year

# Drop the original unixTime column
ChillerData = ChillerData.drop(columns=['Time'])

# Define features and targets
y = ChillerData[['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw']]
X = ChillerData.drop(columns=['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# early stop
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
# Build the neural network model
model = models.Sequential()

# Add layers (adjust neurons and activation functions based on your needs) #, kernel_regularizer=keras.regularizers.l2(0.001)
model.add(layers.Dense(32, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
#model.add(layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))


# Output layer (for regression tasks, use a single neuron with linear activation)
model.add(layers.Dense(5, activation='linear'))


optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=500, callbacks=[early_stopping],  batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")

# Make predictions
y_pred_nn = model.predict(X_test_scaled)


# Plots
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss_PC.png')
plt.show()

plt.plot(history.history['mean_absolute_error'], label='train_MAE')
plt.plot(history.history['val_mean_absolute_error'], label='val_MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()
plt.title('MAE During Training and Validation')
plt.savefig('Accuracy_PC.png')
plt.show()

 
# Evaluate the model for the features in X
mse_deltaP_78 = mean_squared_error(y_test['deltaP_78'], y_pred_nn[:, 0])
r2_deltaP_78 = r2_score(y_test['deltaP_78'], y_pred_nn[:, 0])
mse_deltaP_14 = mean_squared_error(y_test['deltaP_14'], y_pred_nn[:, 1])
r2_deltaP_14 = r2_score(y_test['deltaP_14'], y_pred_nn[:, 1])
mse_deltaP_12 = mean_squared_error(y_test['deltaP_12'], y_pred_nn[:, 2])
r2_deltaP_12 = r2_score(y_test['deltaP_12'], y_pred_nn[:, 2])
mse_Tfw = mean_squared_error(y_test['Tfw'], y_pred_nn[:, 3])
r2_Tfw = r2_score(y_test['Tfw'], y_pred_nn[:, 3])
mse_ffw = mean_squared_error(y_test['ffw'], y_pred_nn[:, 4])
r2_ffw = r2_score(y_test['ffw'], y_pred_nn[:, 4])

resultsNN = {
    'MSE deltaP_78': mse_deltaP_78,
    'R2 deltaP_78': r2_deltaP_78,
    'MSE deltaP_14': mse_deltaP_14,
    'R2 deltaP_14': r2_deltaP_14,
    'MSE deltaP_12': mse_deltaP_12,
    'R2 deltaP_12': r2_deltaP_12,
    'MSE Tfw': mse_Tfw,
    'R2 Tfw': r2_Tfw,
    'MSE ffw': mse_ffw,
    'R2 ffw': r2_ffw
}

resultsNN
print(resultsNN)
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(y_test))

plt.figure(figsize=(16, 9))
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 9), sharex=True)

# Subplot 1: True vs Predicted deltaP_78
ax1.plot(x, y_test['deltaP_78'], 'o-', label='True')
ax1.plot(x, y_pred_nn[:, 0], 'x-', label='Predicted')
ax1.set_ylabel('Pressure (bar)', fontsize=14)
ax1.set_title('True vs Predicted deltaP_78: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_78, r2_deltaP_78))
ax1.legend()

# Subplot 2: True vs Predicted deltaP_14
ax2.plot(x, y_test['deltaP_14'], 'o-', label='True')
ax2.plot(x, y_pred_nn[:, 1], 'x-', label='Predicted')
ax2.set_ylabel('Pressure (bar)', fontsize=14)
ax2.set_title('True vs Predicted deltaP_14: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_14, r2_deltaP_14))
ax2.legend()

# Subplot 3: True vs Predicted deltaP_12
ax3.plot(x, y_test['deltaP_12'], 'o-', label='True')
ax3.plot(x, y_pred_nn[:, 2], 'x-', label='Predicted')
ax3.set_ylabel('Pressure (bar)', fontsize=14)
ax3.set_title('True vs Predicted deltaP_12: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_12, r2_deltaP_12))
ax3.legend()

# Subplot 4: True vs Predicted Tfw
ax4.plot(x, y_test['Tfw'], 'o-', label='True')
ax4.plot(x, y_pred_nn[:, 3], 'x-', label='Predicted')
ax4.set_ylabel('Temperature (°C)', fontsize=14)
ax4.set_title('True vs Predicted Tfw: MSE {:.4f}, R2 {:.4f}'.format(mse_Tfw, r2_Tfw))
ax4.legend()

# Subplot 5: True vs Predicted ffw
ax5.plot(x, y_test['ffw'], 'o-', label='True')
ax5.plot(x, y_pred_nn[:, 4], 'x-', label='Predicted')
ax5.set_xlabel('Sample', fontsize=14)
ax5.set_ylabel('Flow (m$^3$/s)', fontsize=14)
ax5.set_title('True vs Predicted ffw: MSE {:.4f}, R2 {:.4f}'.format(mse_ffw, r2_ffw))
ax5.legend()

plt.suptitle('Performance of Neural Network on Passive Cooler data\n', fontsize=16)
plt.tight_layout()

# Removed explicit x-axis limits for synchronized zooming
plt.savefig('NNtrue_vs_predicted_PC.png')
plt.show()
