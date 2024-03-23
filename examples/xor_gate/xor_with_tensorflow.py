import tensorflow as tf
import numpy as np
import time

start = time.time()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=['mse'],
              metrics=['mse'])

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])



model.fit(inputs, labels, batch_size=0, epochs = 5000)

stop = time.time()

print(model.predict(inputs))
print(stop - start)