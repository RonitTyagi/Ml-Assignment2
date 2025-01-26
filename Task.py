import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the data
independent_file = "logisticX.csv"
dependent_file = "logisticY.csv"

# Handle malformed rows and read data
X = pd.read_csv(independent_file, header=None).values
y = pd.read_csv(dependent_file, header=None).values.flatten()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

# Get coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Cost function (log-loss)
from sklearn.metrics import log_loss
cost_function_value = log_loss(y, model.predict_proba(X_scaled))

print("Cost function value after convergence:", cost_function_value)
print("Learning coefficients:", coefficients)
print("Intercept:", intercept)

# Cost vs iterations plot
costs = []
for i in range(1, 51):
    temp_model = LogisticRegression(solver='lbfgs', max_iter=i)
    temp_model.fit(X_scaled, y)
    temp_cost = log_loss(y, temp_model.predict_proba(X_scaled))
    costs.append(temp_cost)

plt.plot(range(1, 51), costs, marker='o')
plt.title('Cost Function vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.grid()
plt.show()

# Plot dataset and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0][y == 0], X_scaled[:, 1][y == 0], c='red', label='Class 0')
plt.scatter(X_scaled[:, 0][y == 1], X_scaled[:, 1][y == 1], c='blue', label='Class 1')

# Decision boundary
x_values = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_values = -(coefficients[0] * x_values + intercept) / coefficients[1]
plt.plot(x_values, y_values, label='Decision Boundary', color='green')

plt.title('Dataset with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

# Add new variables (squared values)
X_extended = np.hstack([X_scaled, X_scaled ** 2])

# Retrain the model with the extended dataset
model_extended = LogisticRegression(solver='lbfgs', max_iter=1000)
model_extended.fit(X_extended, y)

# Plot extended dataset with decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0][y == 0], X_scaled[:, 1][y == 0], c='red', label='Class 0')
plt.scatter(X_scaled[:, 0][y == 1], X_scaled[:, 1][y == 1], c='blue', label='Class 1')

# Decision boundary for extended dataset
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
grid = np.c_[xx.ravel(), yy.ravel(), (xx ** 2).ravel(), (yy ** 2).ravel()]
z = model_extended.predict(grid)
z = z.reshape(xx.shape)
plt.contour(xx, yy, z, levels=[0.5], colors='green')

plt.title('Extended Dataset with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

# Confusion matrix and metrics
y_pred = model.predict(X_scaled)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
