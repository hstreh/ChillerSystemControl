from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
# Load the dataset
file_pathChiller = r"C:\Users\HETS\Desktop\Helena Skole\Chillerdata.csv"
ChillerData = pd.read_csv(file_pathChiller)

# Convert Time to datetime format
ChillerData['Time'] = pd.to_datetime(ChillerData['Time'], format='%d/%m/%Y %H:%M:%S.%f')

ChillerData['microsecond'] = ChillerData['Time'].dt.microsecond
ChillerData['second'] = ChillerData['Time'].dt.second
ChillerData['minute'] = ChillerData['Time'].dt.minute
ChillerData['hour'] = ChillerData['Time'].dt.hour
ChillerData['day'] = ChillerData['Time'].dt.day
ChillerData['month'] = ChillerData['Time'].dt.month
ChillerData['year'] = ChillerData['Time'].dt.year

# Drop the original unixTime column
ChillerData = ChillerData.drop(columns=['Time'])

# Features and targets
y = ChillerData[['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw']]
X = ChillerData.drop(columns=['deltaP_78', 'deltaP_14', 'deltaP_12', 'Tfw', 'ffw'])


# Add new features based on shifted values of features
X['last_deltaP_78'] = ChillerData['deltaP_78'].shift(1)
X['last_deltaP_14'] = ChillerData['deltaP_14'].shift(1)
X['last_deltaP_12'] = ChillerData['deltaP_12'].shift(1)
X['last_Tfw'] = ChillerData['Tfw'].shift(1)
X['last_ffw'] = ChillerData['ffw'].shift(1)

# Split the data into training and testing sets
test_size = int(len(X) * 0.2)
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

print(y_train.describe())
print(X_train.describe())

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)


# Initialize lists to store the training and validation loss
train_losses = []
val_losses = []
depths = range(1, 40)  #testing tree depths from 1 to 20

# Loop through different tree depths to visualize overfitting/underfitting
for depth in depths:
    # Create and train the model
    model = RandomForestRegressor(max_depth=depth, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_test_scaled)
    
    # Calculate Mean Squared Error for training and validation sets
    train_loss = mean_squared_error(y_train_scaled, y_train_pred)
    val_loss = mean_squared_error(y_test_scaled, y_val_pred)
    
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
plt.savefig('Loss_Tree_Chiller.png')
plt.show()

# Save the model with the best depth
best_depth = depths[np.argmin(val_losses)]
print('Best depth: ', best_depth)
best_model = RandomForestRegressor(max_depth=best_depth, random_state=42)
best_model.fit(X_train_scaled, y_train_scaled)
joblib.dump(best_model, 'random_forest_model_best_depth.pkl')

# Make final predictions with the best model
y_pred = best_model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred)

joblib.dump(model, 'random_forest_modelChiller.pkl')
joblib.dump(y_pred, 'y_pred_forest_modelChiller.pkl')
joblib.dump(X_test, 'X_test_forest_modelChiller.pkl')
joblib.dump(y_test, 'y_test_forest_modelChiller.pkl')

# Evaluate the model
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

plt.suptitle('Performance of Random Forest Regressor on Test Data for Chiller Unit\n', fontsize=16)
plt.tight_layout()

plt.savefig('true_vs_predicted_chiller.png')
plt.show()
