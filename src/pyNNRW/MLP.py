# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # one-hot-vector -> label
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def _change_one_hot_label(X, output_size):
    T = np.zeros((X.size, output_size))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

class MLP:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = x @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = softmax(a2)
        
        return y
        
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def bp_gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = x @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = z1.T @ dy
        grads['b2'] = np.mean(dy, axis=0) # np.sum
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = x.T @ da1
        grads['b1'] = np.mean(da1, axis=0) # np.sum

        return grads
    
    def gradient_check(self, x_batch, t_batch):

        # x_batch = x_train[:5]
        # t_batch = t_train[:5]

        grad_numerical = self.numerical_gradient(x_batch, t_batch)
        grad_backprop = self.bp_gradient(x_batch, t_batch)

        for key in grad_numerical.keys():
            diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
            print(key + ":" + str(diff)) # should be near 0


    def train(self, x_train, t_train, x_test, t_test, 
    iters = 10000, batch_size = 100, learning_rate = 0.1):

        
        train_size = x_train.shape[0]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters):

            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            grad = self.bp_gradient(x_batch, t_batch) # BP is much faster than the numerical gradient method
            
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.params[key] -= learning_rate * grad[key]
            
            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        plt.figure()
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_acc_list))
        plt.plot(x, train_acc_list, label='train acc')
        plt.plot(x, test_acc_list, label='test acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(train_loss_list)), train_loss_list)
        plt.xlabel("iterations")
        plt.ylabel("train loss")
        plt.show()

        return train_loss_list, train_acc_list, test_acc_list

from sklearn.neural_network import MLPClassifier
def create_mlp_instance(L, activation = 'relu'):
    '''
    The above is a self-implementation of MLP. This one reuses the sklearn.neural_network.MLPClassifier.
    For regression problem, don't use this. Should use MLPRegressor.
    
    Parameters
    ----------

    L: tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
        Activation function for the hidden layer.
            ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
            ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
            ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
    '''
    return MLPClassifier(activation = activation, hidden_layer_sizes = L)