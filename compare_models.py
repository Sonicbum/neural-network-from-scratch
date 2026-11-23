import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#1st step make moons dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state = 42)

#2nd step create a model with 5 layers
model = keras.Sequential(
    [
        layers.Dense(8, activation='relu', name='l1'),
        layers.Dense(6, activation='relu', name='l2'),
        layers.Dense(4, activation='relu', name='l3'),
        layers.Dense(2, activation='relu', name='l4'),
        layers.Dense(1, activation='sigmoid', name='output'),

    ]
)


#3rd step compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', #loss function for 2 classes 0 and 1
    metrics=['accuracy'] #to show % of correct predictions
)

model.fit(X,y, epochs=300)

#4th step visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels = 20, cmap='RdBu', alpha = 0.6)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'RdBu', edgecolors = 'black')

plt.title('Neural Network Decision Boundary')
plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.colorbar(label="Predicted class (probability)")

plt.show()