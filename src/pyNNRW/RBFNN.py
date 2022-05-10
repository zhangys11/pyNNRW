'''
Radial basis function network

A RBF SVM would be virtually equivalent to a RBF neural nets where the weights of the first layer would be fixed to the feature values of all the training samples. Only the second layer weights are tuned by the learning algorithm. This allows the optimization problem to be convex hence admit a single global solution. The fact that the number of the potential hidden nodes can grow with the number of samples makes this hypothetical neural network a non-parametric model (which is usually not the case when we train neural nets: we tend to fix the architecture in advance as a fixed hyper-parameter of the algorithm independently of the number of samples).
    RBF neural nets have a higher number of hyper-parameters (the bandwidth of the RBF kernel, number of hidden nodes + the initialization scheme of the weights + the strengths of the regularizer a.k.a weight decay for the first and second layers + learning rate + momentum) + the local optima convergence issues (that may or not be an issue in practice depending on the data and the hyper-parameter)
    RBF SVM has 2 hyper-parameters to grid search (the bandwidth of the RBF kernel and the strength of the regularizer) and the convergence is independent from the init (convex objective function)

A rbf implementation in keras
ref: https://github.com/PetraVidnerova/rbf_keras

𝑅𝐵𝐹(𝑥,𝑥′)=𝑒 (−||𝑥−𝑥′|| / 2𝜎^2 )

参数beta： 𝛽=1 / 2𝜎^2

'''

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant

import math
import numpy as np
from sklearn.cluster import KMeans

class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from sklearn.base import BaseEstimator, ClassifierMixin

class RBFNN(BaseEstimator, ClassifierMixin):
    '''
    A wrapper for RBF-NN.
    Encapsulate as a sklearn-compatible base learner/estimator
    '''
    
    def __init__(self, n_hidden_nodes = 20, flavor = 'classifier'):
        
        # ===============================
        # parameters
        # ===============================
        self.n_hidden_nodes = n_hidden_nodes # RBF node numbers
        self.flavor = flavor # 'classifier' # or 'regressor'

    def fit(self, X, y, batch_size = 16, epochs = 20):
        
        # ===============================
        # Instantiate
        # ===============================
        self.model = Sequential()
        self.rbflayer = RBFLayer(output_dim = self.n_hidden_nodes,
                            initializer = InitCentersRandom(X), # InitCentersKMeans(X)
                            betas=2.0,
                            input_shape=(X.shape[1],))
        self.model.add(self.rbflayer)
        if self.flavor == 'classifier' and len(y.shape) == 2: # y should be one-hot 
            self.model.add(Dense(y.shape[1]))
            num_classes = y.shape[1]
        else:
            num_classes = 2
            self.model.add(Dense(1)) # binary classification or regression

        if self.flavor == 'classifier':
            if num_classes == 2:
                self.model.compile(loss='binary_crossentropy', optimizer=RMSprop())
            else:
                self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
        else:
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())
        
        self.model.fit(X, y, batch_size=batch_size,
          epochs=epochs)

        self.rbf_weights = self.rbflayer.weights
        self.rbf_centers = self.rbflayer.get_weights()[0] # centroids, landmarks
        self.rbf_betas = self.rbflayer.get_weights()[1].tolist()
        self.rbf_sigmas = list( map(lambda x: math.sqrt(x/2.0), self.rbf_betas) )

    #def predict_proba(self, X):        
    #    yh = self.model.predict(X)
    #    # print(yh) # array e.g., 0.37854525 0.         0.         
    #    return yh

    def predict(self, X):
        yh = self.model.predict(X)
        if (len(yh.shape) <= 1):
            return yh > 0.5 # default 0.5 threshold
        yh = np.argmax(yh, axis=-1)
        # print(yh.shape)
        return yh

class RBFNNClassifier(RBFNN):

    def isRBFNN(self):
        return True

    def __init__(self, n_hidden_nodes = 20):

        # invoking the __init__ of the parent class
        RBFNN.__init__(self, n_hidden_nodes, 'classfier')   
        

def create_rbfnn_instance(L, activation = 'relu'):
    '''
    Parameters
    ----------
    L : n_hidden_nodes
    '''
    return RBFNNClassifier(n_hidden_nodes = L)