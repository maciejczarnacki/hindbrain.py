from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import numpy as np

import hindbrain as hb

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

moons_model = hb.Model(name='Moons')

moons_model.add_layer(hb.InputLayer(2))
moons_model.add_layer(hb.LinearLayer(10), activation='tanh')
moons_model.add_layer(hb.LinearLayer(10), activation='tanh')
# moons_model.add_layer(hb.LinearLayer(20), activation='relu')
moons_model.add_layer(hb.LinearLayer(1), activation='sigmoid')

moons_model.build(loss='binary_cross_entropy', optimizer='SGD', learning_rate=0.1   , momentum=0.9)

for i in range(3):
    for x_, y_ in zip(X_train, y_train):
        moons_model.train(x_, y_)

c = 0
for z in range(len(X_test)):
    a = moons_model.predict(X_test[z])
    b = y_test[z]
    if a <= 0.5 and b == 0:
        c += 1
    elif a > 0.5 and b == 1:
        c += 1
    else:
        c += 0
        
print(f'Accuracy: {c/len(X_test)}')

print(len(X_test))