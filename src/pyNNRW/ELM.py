'''
Major revisions: 
1. added multi-class support 
2. Encapsule ELM into a sklearn compatible estimator 

This ELM implementation is based on https://github.com/otenim/Numpy-ELM: 

MIT License

Copyright (c) 2019 Otenim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import numpy as np
import h5py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from . import to_categorical
from sklearn.metrics import log_loss, accuracy_score, recall_score
from sklearn.preprocessing import OneHotEncoder
# from keras import losses

def _mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def _mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _identity(x):
    return x

def _relu(x):
    return np.maximum(0, x)

def softmax(x):
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower

class ELM(object):
    def __init__(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        activation='sigmoid', loss='mean_squared_error', name=None,
        beta_init=None, alpha_init=None, bias_init=None):

        self.name = name
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        # initialize weights and a bias
        if isinstance(beta_init, np.ndarray):
            if beta_init.shape != (self.__n_hidden_nodes, self.__n_output_nodes):
                raise ValueError(
                    'the shape of beta_init is expected to be (%d,%d).' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__beta = beta_init
        else:
            self.__beta = np.random.uniform(-1.,1.,size=(self.__n_hidden_nodes, self.__n_output_nodes))
        if isinstance(alpha_init, np.ndarray):
            if alpha_init.shape != (self.__n_input_nodes, self.__n_hidden_nodes):
                raise ValueError(
                    'the shape of alpha_init is expected to be (%d,%d).' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__alpha = alpha_init
        else:
            self.__alpha = np.random.uniform(-1.,1.,size=(self.__n_input_nodes, self.__n_hidden_nodes))
        if isinstance(bias_init, np.ndarray):
            if bias_init.shape != (self.__n_hidden_nodes,):
                raise ValueError(
                    'the shape of bias_init is expected to be (%d,).' % (self.__n_hidden_nodes,)
                )
            self.__bias = bias_init
        else:
            self.__bias = np.zeros(shape=(self.__n_hidden_nodes,))

        # set an activation function
        self.__activation = self.__get_activation_function(activation)

        # set a loss function
        self.__loss = self.__get_loss_function(loss)

    # FP
    def __call__(self, x):
        h = self.__activation(x.dot(self.__alpha) + self.__bias)
        return h.dot(self.__beta)

    def predict(self, x):
        return np.array(self(x))

    def evaluate(self, x, t, metrics=['loss']):
        y_pred = self.predict(x)
        y_true = t
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        y_true_argmax = np.argmax(y_true, axis=-1)
        ret = []
        for m in metrics:
            if m == 'loss':
                loss = self.__loss(y_true, y_pred)
                ret.append(loss)
            elif m == 'accuracy':
                acc = np.mean(y_pred_argmax == y_true_argmax)
                ret.append(acc)
            elif m == 'precision':
                num_classes = t.shape[1] # len(t[0])
                precision = []
                for i in range(num_classes):
                    tp = np.sum((y_pred_argmax == i) & (y_true_argmax == i))
                    tp_fp = np.sum(y_pred_argmax == i)
                    precision.append(tp / tp_fp)
                ret.append(np.mean(precision))
            elif m == 'recall':
                num_classes = t.shape[1] # len(t[0])
                recall = []
                for i in range(num_classes):
                    tp = np.sum((y_pred_argmax == i) & (y_true_argmax == i))
                    tp_fn = np.sum(y_true_argmax == i)
                    recall.append(tp / tp_fn)
                ret.append(np.mean(recall))
            else:
                raise ValueError(
                    'an unknown evaluation indicator \'%s\'.' % m
                )
        if len(ret) == 1:
            ret = ret[0]
        elif len(ret) == 0:
            ret = None
        return ret

    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            # print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            # print('the label must be reshaped before being transformed')
        return target

    def fit(self, x, t):
        one_hot_target = self._transform_label(t)

        H = self.__activation(x.dot(self.__alpha) + self.__bias)

        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)

        # update beta
        self.__beta = H_pinv.dot(one_hot_target)

    def save(self, filepath):
        with h5py.File(filepath, 'w') as f:
            arc = f.create_dataset('architecture', data=np.array([self.__n_input_nodes, self.__n_hidden_nodes, self.__n_output_nodes]))
            arc.attrs['activation'] = self.__get_activation_name(self.__activation).encode('utf-8')
            arc.attrs['loss'] = self.__get_loss_name(self.__loss).encode('utf-8')
            arc.attrs['name'] = self.name.encode('utf-8')
            f.create_group('weights')
            f.create_dataset('weights/alpha', data=self.__alpha)
            f.create_dataset('weights/beta', data=self.__beta)
            f.create_dataset('weights/bias', data=self.__bias)

    def __get_activation_function(self, name):
        if name == 'sigmoid':
            return _sigmoid
        elif name == 'identity':
            return _identity
        elif name == 'relu':
            return _relu
        else:
            raise ValueError(
                'an unknown activation function \'%s\'.' % name
            )

    def __get_activation_name(self, activation):
        if activation == _sigmoid:
            return 'sigmoid'
        elif activation == _identity:
            return 'identity'

    def __get_loss_function(self, name):
        if name == 'mean_squared_error':
            return _mean_squared_error
        elif name == 'mean_absolute_error':
            return _mean_absolute_error
        else:
            raise ValueError(
                'an unknown loss function \'%s\'.' % name
            )

    def __get_loss_name(self, loss):
        if loss == _mean_squared_error:
            return 'mean_squared_error'
        elif loss == _mean_absolute_error:
            return 'mean_absolute_error'
    
    @property
    def weights(self):
        return {
            'alpha': self.__alpha,
            'beta': self.__beta,
            'bias': self.__bias,
        }

    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    @property
    def output_shape(self):
        return (self.__n_output_nodes,)

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes

    @property
    def activation(self):
        return self.__get_activation_name(self.__activation)

    @property
    def loss(self):
        return self.__get_loss_name(self.__loss)

def load_model(filepath):
    with h5py.File(filepath, 'r') as f:
        alpha_init = f['weights/alpha'][...]
        beta_init = f['weights/beta'][...]
        bias_init = f['weights/bias'][...]
        arc = f['architecture']
        n_input_nodes = arc[0]
        n_hidden_nodes = arc[1]
        n_output_nodes = arc[2]
        activation = arc.attrs['activation'].decode('utf-8')
        loss = arc.attrs['loss'].decode('utf-8')
        name = arc.attrs['name'].decode('utf-8')
        model = ELM(
            n_input_nodes=n_input_nodes,
            n_hidden_nodes=n_hidden_nodes,
            n_output_nodes=n_output_nodes,
            activation=activation,
            loss=loss,
            alpha_init=alpha_init,
            beta_init=beta_init,
            bias_init=bias_init,
            name=name,
        )
    return model


class ELMClassifierCV():

    def __init__(self, hparams = {'n_hidden_nodes': [1, 10], 'activation': ['relu', 'sigmoid', 'identity'] }):        
        '''
        parameters  
        ----------
        hparams : hyper-parameter candidate values to be grid-searched
        '''
        self.parameters =hparams
       
    def fit(self, X, y):
        elmc = ELMClassifier()
        self.clf = GridSearchCV(elmc, self.parameters, scoring = 'accuracy') # 'neg_log_loss'
        self.clf.fit(X, y)
        self.classes_ = np.unique(y)
        # print( sorted(clf.cv_results_.keys()) )
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)


class ELMClassifier(BaseEstimator, ClassifierMixin):
    '''
    A wrapper for ELM.
    Encapsulate as a sklearn-compatible base learner/estimator
    '''
    
    def __init__(self, n_hidden_nodes = 20, activation = 'relu'):
        
        # ===============================
        # ELM parameters
        # ===============================
        self.model = None # will be instantiated in fit()
        self.classes_ = 0 # will be instantiated in fit()
        self.n_features_in_ = 0 # will be instantiated in fit()

        self.n_hidden_nodes = n_hidden_nodes #x_train.shape[1]
        self.loss = 'mean_squared_error' # 'mean_absolute_error'
        self.activation = activation # 'sigmoid' # 'identity' # 'relu'
        
    def fit(self, X, y):
        '''
        Instantiate an inner ELM and fit it to the data
        '''
        
        if len(y.shape) == 2: #one-hot
            n_classes = y.shape[1]
        else:
            n_classes = len(set(y))

        self.model = ELM(
            n_input_nodes = X.shape[1],
            n_hidden_nodes = self.n_hidden_nodes,
            n_output_nodes = n_classes,
            loss = self.loss,
            activation = self.activation,
            name = 'elm'
        )
        
        self.classes_ = np.unique(y)
        self.model.fit(X, y)
        self.n_features_in_ = X.shape[1] # n_features_in_ is the number of features that an estimator expects.

    def predict_proba(self, X):
        '''
        This doesn't return the probability directly, but the raw output from the ann.
        You may call softmax() on the output to get the formalized probability. 
        '''
        yh = self.model.predict(X) # this is a direct output from ann, e.g., 0.37854525 0.         0.         
        return yh

    def predict(self, X):
        '''
        return the class labels
        '''
        yh = self.model.predict(X) # (m,  n_classes)
        if (len(yh.shape) <= 1): # e.g., (m, )
            return yh > 0.5 # default 0.5 threshold
        yh = np.argmax(yh, axis=-1)
        return yh

    def evaluate(self, val_x, val_y, metrics=['loss', 'accuracy']):

        if len(val_y.shape) == 2: #one-hot
            n_classes = val_y.shape[1]
            y_gt = val_y
            val_y = val_y.argmax(-1)
        else:
            n_classes = len(set(val_y))
            y_gt = to_categorical(val_y, len(set(val_y))).astype(np.float32)
        
        y_hat = self.predict_proba(val_x)
        y_pred = self.predict(val_x)
        # print(val_y.shape, y_gt.shape, y_hat.shape, y_pred.shape)
        
        val_loss = log_loss(y_gt, y_hat)
        val_acc = accuracy_score(val_y, y_pred)
        val_recall = recall_score(val_y, y_pred, average = 'macro')
        
        return val_loss, val_acc

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def run_example(x_train, x_test, t_train, t_test, save_model_path = ''):
        
        # ===============================
        # ELM parameters
        # ===============================
        n_hidden_nodes = x_train.shape[1]
        loss = 'mean_squared_error' # 'mean_absolute_error'
        activation = 'sigmoid' # 'identity'
        n_classes = t_train.shape[1]

        # ===============================
        # Instantiate ELM
        # ===============================
        clf = ELMClassifier()

        # ===============================
        # Training
        # ===============================
        clf.fit(x_train, t_train)
        train_loss, train_acc = clf.evaluate(x_train, t_train, metrics=['loss', 'accuracy'])
        print('train_loss: %f' % train_loss) # loss value
        print('train_acc: %f' % train_acc) # accuracy

        # ===============================
        # Validation
        # ===============================
        val_loss, val_acc = clf.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
        print('val_loss: %f' % val_loss)
        print('val_acc: %f' % val_acc)

        # ===============================
        # Prediction
        # ===============================
        print("\n\n========== prediction on the first 10 test samples ===========\n")
        x = x_test[:10]
        t = t_test[:10]
        y_pred = clf.predict_proba(x)

        for i, y_pred_i in enumerate(y_pred):
            print('---------- prediction %d ----------' % (i+1))
            class_pred = np.argmax(y_pred_i)
            prob_pred = y_pred_i[class_pred]
            class_true = np.argmax(t[i])
            print('class: %d, probability: %f' % (class_pred, prob_pred))
            print('class (true): %d' % class_true)

        if (save_model_path):

            # ===============================
            # Save model
            # ===============================
            clf.save_model(save_model_path)

            # ===============================
            # Load model
            # ===============================
            # clf.load_model(save_model_path)

    def run_iris_example():

        from sklean.datasets import load_iris

        # ===============================
        # Load dataset
        # ===============================
        iris = load_iris()
        n_classes = len(set(iris.target))
        # stdsc = StandardScaler()
        # irisx = stdsc.fit_transform(iris.data)
        x_train, x_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size=0.2)
        t_train = to_categorical(t_train, n_classes).astype(np.float32)
        t_test = to_categorical(t_test, n_classes).astype(np.float32)

        ELMClassifier.run_example(x_train, x_test, t_train, t_test)

    def run_mnist_example():
        
        from keras.datasets import mnist

        n_classes = 10
        (x_train, t_train), (x_test, t_test) = mnist.load_data()

        # ===============================
        # Preprocess
        # ===============================
        x_train = x_train.astype(np.float32) / 255.
        x_train = x_train.reshape(-1, 28**2)
        x_test = x_test.astype(np.float32) / 255.
        x_test = x_test.reshape(-1, 28**2)
        t_train = to_categorical(t_train, n_classes).astype(np.float32)
        t_test = to_categorical(t_test, n_classes).astype(np.float32)

        ELMClassifier.run_example(x_train, x_test, t_train, t_test)

def create_elm_instance(L, activation = 'relu'):
    '''
    Parameters
    ----------
    L : n_hidden_nodes
    activation : {'relu', 'sigmoid', 'identity'}
    '''
    return ELMClassifier(n_hidden_nodes = L, activation = activation) # 'identity'


def create_elmcv_instance():
    return ELMClassifierCV()

def ElmRegression(x_train, x_test, t_train, t_test, L = 10, rndb = False):
    
    # ===============================
    # ELM parameters
    # ===============================
    n_input_nodes = x_train.shape[1]
    n_hidden_nodes = max(x_train.shape[1], L) # at least 10 hidden nodes
    n_output_nodes = t_train.shape[1]

    # ===============================
    # Initialization
    # ===============================
    A = np.random.uniform(-1.,1.,size=(n_input_nodes, n_hidden_nodes))
    if rndb:
        b = np.random.uniform(-1.,1.,size=(n_hidden_nodes,))
    else:
        b = np.zeros(shape=(n_hidden_nodes,))

    # ===============================
    # Solve
    # ===============================
    H = _sigmoid(x_train @ A + b)
    B = np.linalg.pinv(H) @ t_train

    # ===============================
    # Evaluate on CV dataset
    # ===============================
    H = _sigmoid(x_test @ A + b)
    t_pred = H @ B

    MSE = np.mean((t_test - t_pred)**2)
    return MSE, A, b, B