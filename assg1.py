import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate input and output
X_train = train_data[['x']]
y_train = train_data['y']

X_test = test_data[['x']]
y_test = test_data['y']

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Display model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)

# Generate regression line
x_line = np.linspace(
    X_train['x'].min(),
    X_train['x'].max(),
    200
).reshape(-1, 1)

x_line = pd.DataFrame(x_line, columns=['x'])
y_line = model.predict(x_line)

# Plot 1: Training data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_train['x'], y_train, color='blue',
            label='Training Data')
plt.plot(x_line['x'], y_line, color='red',
         label='Regression Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression - Training Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Actual vs Predicted testing values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green',
            label='Predicted Values')

# Perfect prediction reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val],
         [min_val, max_val],
         color='red', label='Perfect Prediction')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()