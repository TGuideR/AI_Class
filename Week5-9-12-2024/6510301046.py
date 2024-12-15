import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd
import tensorflow as tf
import random

# Set random seeds
random.seed(69)
np.random.seed(69)
tf.random.set_seed(69)


# Make Dataset
# Dataset 1
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0, 2.0),
                    cluster_std=0.75,
                    random_state=69)

# Dataset 2
x2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0, 3.0),
                    cluster_std=0.75,
                    random_state=69)

# Label Output
y2 = np.ones(y2.shape[0])

# Combine datasets and output
X = np.vstack((x1, x2))
y = np.hstack((y1, y2))

#print(X.shape, y.shape)

# Plot Data
# plt.scatter(X[:100, 0], X[:100, 1], label='class 1')
# plt.scatter(X[101:, 0], X[101:, 1], label='class 2')
# plt.legend()
# plt.xlabel('x1 Features')
# plt.ylabel('x2 Features')
# plt.show()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a neural network model
model = Sequential()
model.add(Dense(1, activation='linear', input_dim=2))
optimizer = Adam(0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=300, verbose=0)

# Making predictions
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).astype(int).ravel()

# Plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'])
plt.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2)
plt.scatter(X[:100, 0], X[:100, 1], c='red', linewidths=1, alpha=0.6, label="Class 1", edgecolors='k')
plt.scatter(X[101:, 0], X[101:, 1], c='blue', linewidths=1, alpha=0.6, label="Class 2", edgecolors='k')
plt.xlabel('x1 Features')
plt.ylabel('x2 Features')
plt.title('Decision Boundary')
plt.show()