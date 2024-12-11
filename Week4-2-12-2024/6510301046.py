from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Define decision function
def decision_function(x1, x2, weights: np.array):
    return weights[1]*x1 + weights[2]*x2 + 1*weights[0]

def activation_function(z: np.array):
    return np.where(z <= 0, 0, 1)

def cost_function(X, y, predict):
    return abs(sum((predict - y))) / X.shape[0]

def train(X, y, learning_rate, epoch, weights, predict):
    for i in range(epoch):
        update_weights = weights - np.sum((learning_rate / np.shape(X)[0]) * X.transpose() * (predict-y))
        weights = update_weights
    return weights


# Dataset 1
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0, 2.0),
                    cluster_std=0.25,
                    random_state=69)

# Dataset 2
x2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0, 3.0),
                    cluster_std=0.25,
                    random_state=69)

# Label Output y
y2 = np.ones(y2.shape[0])

# Combine datasets and output
X = np.vstack((x1, x2))
y = np.hstack((y1, y2))

# print(X, y)

# Initial weights
weights = np.array([0.5, 1, 1])
g_values = decision_function(X[:, 0], X[:, 1], weights)
z = activation_function(g_values)
cost = cost_function(X, y, z)
print(f"w0, w1, w2 = {weights}")
print(f"Error = {cost * 100} %")

# Adjust weights
weights = train(X, y, 0.5, 5, weights, z)
g_values = decision_function(X[:, 0], X[:, 1], weights)
z = activation_function(g_values)
cost = cost_function(X, y, z)
print(f"w0, w1, w2 = {weights}")
print(f"Error = {cost * 100} %")

# Create a grid for contour
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                               np.linspace(x2_min, x2max, 200))

g_values = decision_function(x1_grid, x2_grid, weights)


# Plot
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.6)
plt.scatter(x1[:, 0], x1[:, 1], c='red', linewidths=1, alpha=0.6, label="Class 1")
plt.scatter(x2[:, 0], x2[:, 1], c='blue', linewidths=1, alpha=0.6, label="Class 2")

# Plot decision boundary (contour where g_values = 0)
plt.contour(x1_grid, x2_grid, g_values, colors='black', levels=[0], linewidths=2)

plt.title("Decision Boundary")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.grid()
plt.legend()
plt.show()
