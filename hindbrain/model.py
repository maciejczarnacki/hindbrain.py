import numpy as np
import pickle
from .layers import InputLayer, LinearLayer
from .activations import (relu, 
                          d_relu, 
                          sigmoid, 
                          d_sigmoid, 
                          linear, 
                          d_linear, 
                          softmax, 
                          d_softmax,
                          softmax_b,
                          d_softmax_b, 
                          tanh, 
                          d_tanh, 
                          elu, 
                          d_elu)
from .losses import (mse, 
                     d_mse, 
                     mae, 
                     d_mae, 
                     categorical_cross_entropy, 
                     d_categorical_cross_entropy, 
                     binary_cross_entropy, 
                     d_binary_cross_entropy)
from .optimizers import SGD, RMSpopr, Adam, Amsgrad
from .initializers import weights_init, biases_init
from .tools import one_hot, flatten, accuracy

class Model:
    def __init__(self, name: str ='Default_model') -> None:
        self.name = name
        self.model = []
        self.forward_passed = False
        self.weights = None
        self.biases = None
        self.loss = None
        self.history = []
        self.loss_value = None
    
    def add_layer(self, 
                  layer: InputLayer|LinearLayer, 
                  activation: str = 'None') -> None:
        layer_number = len(self.model)
        self.model.append((layer_number, layer, activation))
    
    def build(self, 
              loss: str = 'mse', 
              optimizer: str ='SGD', 
              learning_rate: float = 0.001, 
              momentum: float = 0.9, 
              beta: float = 0.999, 
              epsilon: float = 10**(-7), 
              initializer: str ='he_normal') -> None:
        # generate weights and biases
        self.loss = loss
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.epsilon = epsilon
        sizes = []
        self.input_size = self.model[0][1].size
        for number, layer, activation in self.model:
            sizes.append(layer.size)
        self.weights = [weights_init(shape=(x, y),initializer=initializer)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [biases_init(shape=(y, 1)).reshape(1, y)
                       for y in sizes[1:]]
        for i, (number, layer, activation) in enumerate(self.model[1:]):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]
        # chose optimizer (SGD, RMSprop, Adam)
        self.choose_optimizer(optimizer)
         
    def activation_function(self, 
                            activation: str, 
                            x: np.ndarray) -> np.ndarray:
        if activation == 'relu': output = relu(x)
        elif activation == 'elu': output = elu(x)
        elif activation == 'softmax': output = softmax(x)
        elif activation == 'sigmoid': output = sigmoid(x)
        elif activation == 'linear': output = linear(x)
        elif activation == 'tanh': output = tanh(x)
        else: 
            print('Unknown activation function')
            return None
        return output
    
    def activation_function_derivative(self, 
                                       activation: str, 
                                       x: np.ndarray) -> np.ndarray:
        if activation == 'relu': output = d_relu(x)
        elif activation == 'elu': output = d_elu(x)
        elif activation == 'softmax': output = d_softmax(x)
        elif activation == 'sigmoid': output = d_sigmoid(x)
        elif activation == 'linear': output = d_linear(x)
        elif activation == 'tanh': output = d_tanh(x)
        else: 
            print('Unknown activation function')
            return None
        return output
    
    def choose_optimizer(self, optimizer: str) -> None:
        if optimizer == 'SGD': self.optimizer = SGD(self.learning_rate, 
                                                    self.momentum)
        elif optimizer == 'RMSprop': self.optimizer = RMSpopr(self.learning_rate, 
                                                              self.beta, 
                                                              self.epsilon)
        elif optimizer == 'Adam': self.optimizer = Adam(self.learning_rate,
                                                        self.momentum, 
                                                        self.beta,
                                                        self.epsilon)
        elif optimizer == 'AMSgrad': self.optimizer = Amsgrad(self.learning_rate,
                                                self.momentum, 
                                                self.beta,
                                                self.epsilon)
        else:
            raise ValueError('Unknown optimizer, choose from SGD, RMSprop, Adam or AMSgrad.')
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_ = np.copy(x)
        if self.model != None:
            for number, layer, activation in self.model:
                x_ = layer.forward(x_)
                if layer.layer_name !='input_layer':
                    x_ = self.activation_function(activation, x_)
                    layer.forward_activation_output = x_
            self.forward_passed = True
            return x_

    #train methode without batches
    def train(self, x: np.ndarray, labels: np.ndarray, epochs: int = 1) -> None: 
        if self.model != None:
            for ep in range(epochs):
                preds = self.predict(x)
                # backpropagation steps layer by layer
                if self.loss == 'mse': 
                    loss = mse(preds, labels)
                    dx = d_mse(preds, labels)
                elif self.loss == 'mae': 
                    loss = mae(preds, labels)
                    dx = d_mae(preds, labels)
                elif self.loss == 'categorical_cross_entropy':
                    loss = categorical_cross_entropy(labels, preds)
                    dx = d_categorical_cross_entropy(labels, preds)
                elif self.loss == 'binary_cross_entropy':
                    loss = binary_cross_entropy(labels, preds)
                    dx = d_binary_cross_entropy(labels, preds)
                else: 
                    print('Unknown loss function')
                    return None
                # print('loss', round(loss, 6))
                self.loss_value = loss
                for i, (number, layer, activation) in enumerate(reversed(self.model[1:])):
                    if number == len(self.model) - 1:
                        if self.loss == 'categorical_cross_entropy':
                            delta = np.sum(dx * self.activation_function_derivative(activation, layer.forward_output), axis=1).reshape((-1, 1)).T
                        else:
                            dA = self.activation_function_derivative(activation, layer.forward_output)
                            delta = dx * dA
                        dw = np.dot(layer.forward_input.T, delta)
                        db = 1 * delta
                        self.optimizer.update(layer, dw, db)
                    elif number != 1:
                        w_T = self.model[number+1][1].weights.T
                        dA = self.activation_function_derivative(activation, layer.forward_output)
                        dx = np.dot(delta, w_T)
                        delta = dx * dA
                        dw = np.dot(layer.forward_input.T, delta)
                        db = 1 * delta
                        self.optimizer.update(layer, dw, db)
                   

    # work with batches
    """ def train(self, x, labels, epochs=1, batch_size = 1):
        if self.model != None:
            # generate weights
            n = len(x)
            print(x.shape, labels.shape)
            for epoch in range(epochs):
                mini_batches_x = np.asarray([x[step:step+batch_size] for step in range(0, n, batch_size)])
                mini_batches_labels = [labels[step:step+batch_size] for step in range(0, n, batch_size)]
            # for epoch in range(epochs):
                for mini_batch_x, mini_batch_labels in zip(mini_batches_x, mini_batches_labels):
                    preds = self.predict(mini_batch_x)
                    if self.loss == 'mse': 
                        loss = mse(preds, mini_batch_labels)
                        loss = loss
                        dx = d_mse(preds, mini_batch_labels)
                    elif self.loss == 'mae': 
                        loss = mae(preds, mini_batch_labels)
                        loss = loss.mean()
                        dx = d_mae(preds, mini_batch_labels)
                    elif self.loss == 'categorical_cross_entropy':
                        loss = categorical_cross_entropy(mini_batch_labels, preds)
                        # loss = loss.mean()
                        dx = d_categorical_cross_entropy(mini_batch_labels, preds)
                        # print(dx)
                        #dx = d_mse(preds, labels)
                    else: 
                        print('Unknown loss function')
                        return None
                    print('loss', round(loss, 6), 'epoch: ', epoch)
                    # print(dx)
                    for i, (number, layer, activation) in enumerate(reversed(self.model[1:])):
                        if number == len(self.model) - 1:
                            if self.loss == 'categorical_cross_entropy':
                                # print(self.activation_function_derivative(activation, layer.forward_output))
                                # zmieiam na pętlę
                                delta= []
                                for ddx, f_out in zip(dx, layer.forward_output):
                                    delta.append(np.sum(ddx * self.activation_function_derivative(activation, f_out), axis=1).reshape((-1, 1)).T)
                                    # delta = (np.sum(dx * self.activation_function_derivative(activation, layer.forward_output), axis=1).reshape((-1, 1)).T)
                                delta = np.array(delta).squeeze(axis=1)
                                # print(delta, delta.squeeze(a))
                            else:
                                dA = self.activation_function_derivative(activation, layer.forward_output)
                                delta = dx * dA
                            dw = np.dot(layer.forward_input.T, delta)
                            db = 1 * delta
                            # for mdw, mdb in zip(dw, db):
                            for mw in dw:
                                layer.weights -= self.learning_rate * mw
                            for mb in db:
                                layer.biases -= self.learning_rate * mb
                            # print('dw', dw, '\ndb',db)
                            # dw = dw.sum(axis=0)/batch_size
                            # db = db.sum(axis=0)/batch_size
                            # layer.weights -= self.learning_rate * dw
                            # layer.biases -= self.learning_rate * db
                            # layer.vw = ((1 - self.momentum) * dw + self.momentum * layer.vw)
                            # layer.vb = ((1 - self.momentum) * db + self.momentum * layer.vb)
                            # for lvw, lvb in zip(layer.vw, layer.vb):
                            # for step_dw, step_db in zip(dw, db):
                            # step_dw = dw.mean()
                            # step_db = db.mean()
                                # layer.vw = (1 - self.momentum) * step_dw + self.momentum * layer.vw
                                # layer.vb = (1 - self.momentum) * step_db + self.momentum * layer.vb
                                # layer.weights -= self.learning_rate * layer.vw
                                # layer.biases -= self.learning_rate * layer.vb
                        elif number != 1:
                            w_T = self.model[number+1][1].weights.T
                            dA = self.activation_function_derivative(activation, layer.forward_output)
                            dx = np.dot(delta, w_T)
                            delta = dx * dA
                            dw = np.dot(layer.forward_input.T, delta)
                            db = 1 * delta
                            # for mdw, mdb in zip(dw, db):
                            for mw in dw:
                                layer.weights -= self.learning_rate * mw
                            for mb in db:
                                layer.biases -= self.learning_rate * mb
                            # dw = dw.sum(axis=0)/batch_size
                            # db = db.sum(axis=0)/batch_size
                            # layer.weights -= self.learning_rate * dw
                            # layer.biases -= self.learning_rate * db
                            # layer.vw = ((1 - self.momentum) * dw + self.momentum * layer.vw)
                            # layer.vb = ((1 - self.momentum) * db + self.momentum * layer.vb)
                            # step_dw = dw.mean()
                            # step_db = db.mean()
                            # for lvw, lvb in zip(layer.vw, layer.vb):
                            # for step_dw, step_db in zip(dw, db):
                            #     layer.vw = (1 - self.momentum) * step_dw + self.momentum * layer.vw
                            #     layer.vb = (1 - self.momentum) * step_db + self.momentum * layer.vb
                            #     layer.weights -= self.learning_rate * layer.vw
                            #     layer.biases -= self.learning_rate * layer.vb
 """

    def summary(self) -> str:
        """
        Return and print out model structure and learning parameters like learning rate, optimizer, momentum etc...
        """
        self.summary_text = (f'Model name: {self.name}\n'
                        + f'Total number of layers: {len(self.model)}\n'
                        + f'Learning parameters:\n'
                        + f'Loss: {self.loss}, optimizer: {self.optimizer.name}, learning rate: {self.learning_rate}, momentum: {self.momentum}, beta: {self.beta}\n'
                        + f'Model structure\n')
        for number, layer, activation in self.model:
            self.summary_text += f'Layer number: {number}, layer name: {layer.layer_name}, number of neurons: {layer.size}, activation: {activation}\n'
        print(self.summary_text)
        return self.summary_text


def save_model(path: str, model: Model) -> None:
    """
    Save hindbrian model to the file using pickle package.

    Parameters
    -----------
    path: string with file path for pickling data

    model: class 'Model <hindbrain.Model>' - data for pickling
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)
            
def load_model(path: str) -> Model:
    """
    Load hindbrian model from the file using pickle package.

    Parameters
    -----------
    path: string with file path to pickled data

    Returns
    ----------
    model: class 'Model <hindbrain.Model>'
    """
    with open(path, 'rb') as file:
        return pickle.load(file)
            

