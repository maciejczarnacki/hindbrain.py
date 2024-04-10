# hindbrain.py

## Introduction and motivation

During my experiments with creating a Pong game in which the computer learns the game from a human, I noticed that Tensorflow is poorly suited for such a trivial task.
Tensorflow is very inefficient when we train the network on single data in a loop instead of on large data sets organized in packages.
Tensorflow caused a single step of the game loop to last several dozen milliseconds.
However, the vector describing the game state in each frame consisted of five numbers, and the neural network consisted of a maximum of 33 neurons organized in 4 layers (Dense).
At first I thought it was my code's fault, then it was the computer's fault.
After changing the version of the Tensorflow library several times, I came to the conclusion that I needed to create my own basic library. And that's how **hindbrain.py** was created.

## hindbrain.py package structure

**hindbrain.py** is a simple library based on Numpy.
With its help, we can create a model of a neural network consisting of any number of layers containing any number of neurons.
Currently, it is possible to construct linear networks (so-called dense networks), but the work is ongoing and in the near future I will expand it with the possibility of constructing convolutional neural networks.
We have a number of activation functions at our disposal: linear function, hyperbolic tangent, sigmoid, relu, elu, softmax.
There are three loss functions available: mean squared error, mean absolute error, binary and categorical cross entropy.
Currently, three methods for optimizing (training) the network are available: Stochastic Gradient Descent with momentum,
Root Mean Squared Propagation and one variation of the Adam - Adaptive Moment Estimation  optimizer AMSGrad.

**Let's see some code.**

1. Importing hindbrain package

```python
import hindbrain as hb
```

2. Neural network model building

```python
my_model = hb.Model()
my_model.add_layer(hb.InputLayer(2))
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(1), 'sigmoid')

my_model.build(loss='mse', optimizer='SGD', learning_rate=0.15, momentum=0.9, initializer='he_normal')
```

3. Neural network learning in a loop

```python
for epoch in range(500):
    for input, label in zip(inputs, labels):
        my_model.train(input, label)
```

4. Getting predictions from learned model

```python
pred = my_model.predict(input)
```

5. Printing information about model

```python
my_model.summary()
```

6. Saving/loading of learned model to file

```python
file_path = 'my_model_sigmoid'
hb.save_model(file_path, my_model)

loaded_model = hb.load_model(file_path)
```

7. Examples

In the repository I include two basic examples of using **hindbrain.py**.
A model solving the XOR gate problem and a model reading handwritten digits 0-9 trained and tested using the MNIST database.
A repository and an article about Pong are in preparation.

## To do list

1. Batch version of nn algorithm
2. <s>More optimizers (Adam, momentum, etc...)</s> - done
3. Add convolutional layers
4. Build data type manager for models
5. Write good documentation
6. Add recurent neural networks
7. Write a few excamples with benchmarks
8. Parallel computation module

## Done

05.04.2024 -    I have added optimizers - SGD with momentum, RMSprop and Adam.
                Now it is possible to choose weights initializer (Glorot or He normal and uniform)

## Licence

The author of the repository is not responsible for any damage caused using it.
**hindbrain.py** is completely free and available for general use under the MIT license.