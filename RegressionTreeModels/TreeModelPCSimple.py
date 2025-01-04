from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the dataset
file_pathPC = r"PCdata.csv"
PCData = pd.read_csv(file_pathPC)


# Define features and targets
y = PCData[['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw']]
X = PCData[['vfw_1', 'vfw_2', 'vfw_3', 'vsw_1', 'vsw_2','Valve_sw2','Tsw_PC','interval','elapsed_time']]

# Add new features based on shifted values of features
X['last_deltaP_78'] = PCData['deltaP_78'].shift(1)
X['last_deltaP_14'] = PCData['deltaP_14'].shift(1)
X['last_deltaP_12'] = PCData['deltaP_12'].shift(1)
X['last_Tfw'] = PCData['Tfw'].shift(1)
X['last_ffw'] = PCData['ffw'].shift(1)

# Save X to a CSV file for inspection
X.to_csv('X_features_simple.csv', index=False)
print("Feature matrix X saved to 'X_features.csv'")
y.to_csv('y_targets_simple.csv', index=False)
print("Feature matrix X saved to 'X_features.csv'")

# Split the data into training and testing sets
test_size = int(len(X) * 0.2)
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

# Save feature names
feature_names = X_train.columns 
feature_names_file = 'feature_names_simple.csv'
pd.DataFrame(feature_names, columns=["Feature Names"]).to_csv(feature_names_file, index=False)
print(f"Feature names saved to {feature_names_file}")

# Scale the input features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the output/target variables
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Save the scalers
joblib.dump(scaler_X, 'scaler_X_simple.pkl')
joblib.dump(scaler_y, 'scaler_y_simple.pkl')


# Initialize lists to store
train_losses = []
val_losses = []
depths = range(1, 40)  #testing tree depths from 1 to 20

# Loop through different tree depths to visualize overfitting/underfitting
for depth in depths:
    # train the model
    model = RandomForestRegressor(max_depth=depth, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_test_scaled)
    
    # Calculate Mean Squared Error for training and validation sets
    train_loss = mean_squared_error(y_train_scaled, y_train_pred)
    val_loss = mean_squared_error(y_test_scaled, y_val_pred)
    
    # Store the losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot the training and validation loss over different depths
plt.figure(figsize=(8, 6))
plt.plot(depths, train_losses, label='Train Loss', marker='o')
plt.plot(depths, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Mean Squared Error')
plt.title('Train vs Validation Loss for Random Forest Regressor')
plt.legend()
plt.grid(True)
plt.savefig('Loss_Tree_Chiller_simple.png')
plt.show()

# Save the model with the best depth
best_depth = depths[np.argmin(val_losses)]
print('Best depth: ', best_depth)
best_model = RandomForestRegressor(max_depth=best_depth, random_state=42)
best_model.fit(X_train_scaled, y_train_scaled)
joblib.dump(best_model, 'random_forest_model_best_depth_simple.pkl')

# Make final predictions with the best model
y_pred = best_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred)

joblib.dump(model, 'random_forest_modelPC_simple.pkl')
joblib.dump(y_pred, 'y_pred_forest_modelPC_simple.pkl')
joblib.dump(X_test, 'X_test_forest_modelPC_simple.pkl')
joblib.dump(X_train, 'X_train_forest_modelPC_simple.pkl')
joblib.dump(y_test, 'y_test_forest_modelPC_simple.pkl')


# Evaluate the model for the features in X
mse_deltaP_78 = mean_squared_error(y_test['deltaP_78'], y_pred[:, 0])
r2_deltaP_78 = r2_score(y_test['deltaP_78'], y_pred[:, 0])
mse_deltaP_14 = mean_squared_error(y_test['deltaP_14'], y_pred[:, 1])
r2_deltaP_14 = r2_score(y_test['deltaP_14'], y_pred[:, 1])
mse_deltaP_12 = mean_squared_error(y_test['deltaP_12'], y_pred[:, 2])
r2_deltaP_12 = r2_score(y_test['deltaP_12'], y_pred[:, 2])
mse_Tfw = mean_squared_error(y_test['Tfw'], y_pred[:, 3])
r2_Tfw = r2_score(y_test['Tfw'], y_pred[:, 3])
mse_ffw = mean_squared_error(y_test['ffw'], y_pred[:, 4])
r2_ffw = r2_score(y_test['ffw'], y_pred[:, 4])

resultsTree = {
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

resultsTree
print(resultsTree)

x = np.arange(len(y_test))

plt.figure(figsize=(16, 9))
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 9), sharex=True)
# Subplot 1: True vs Predicted deltaP_78
ax1.plot(x, y_test['deltaP_78'], 'o-', label='True')
ax1.plot(x, y_pred[:, 0], 'x-', label='Predicted')
ax1.set_ylabel('Pressure (bar)', fontsize=14)
ax1.set_title('True vs Predicted deltaP_78: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_78, r2_deltaP_78))
ax1.legend()

# Subplot 2: True vs Predicted deltaP_14
ax2.plot(x, y_test['deltaP_14'], 'o-', label='True')
ax2.plot(x, y_pred[:, 1], 'x-', label='Predicted')
ax2.set_ylabel('Pressure (bar)', fontsize=14)
ax2.set_title('True vs Predicted deltaP_14: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_14, r2_deltaP_14))
ax2.legend()

# Subplot 3: True vs Predicted deltaP_12
ax3.plot(x, y_test['deltaP_12'], 'o-', label='True')
ax3.plot(x, y_pred[:, 2], 'x-', label='Predicted')
ax3.set_ylabel('Pressure (bar)', fontsize=14)
ax3.set_title('True vs Predicted deltaP_12: MSE {:.4f}, R2 {:.4f}'.format(mse_deltaP_12, r2_deltaP_12))
ax3.legend()

# Subplot 4: True vs Predicted Tfw
ax4.plot(x, y_test['Tfw'], 'o-', label='True')
ax4.plot(x, y_pred[:, 3], 'x-', label='Predicted')
ax4.set_ylabel('Temperature (°C)', fontsize=14)
ax4.set_title('True vs Predicted Tfw: MSE {:.4f}, R2 {:.4f}'.format(mse_Tfw, r2_Tfw))
ax4.legend()

# Subplot 5: True vs Predicted ffw
ax5.plot(x, y_test['ffw'], 'o-', label='True')
ax5.plot(x, y_pred[:, 4], 'x-', label='Predicted')
ax5.set_xlabel('Sample', fontsize=14)
ax5.set_ylabel('Flow (m$^3$/s)', fontsize=14)
ax5.set_title('True vs Predicted ffw: MSE {:.4f}, R2 {:.4f}'.format(mse_ffw, r2_ffw))
ax5.legend()

plt.suptitle('Performance of Random Forest Regressor on Test Data for Passive Cooler\n', fontsize=16)
plt.tight_layout()

# Remove manual x-axis limits setting to enable dynamic zooming
plt.savefig('true_vs_predicted_PC_simple.png')
plt.show()