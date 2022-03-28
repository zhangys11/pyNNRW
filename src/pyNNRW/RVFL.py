'''
This RVFL implementation is based on RVFL_plus ( https://github.com/pablozhang/RVFL_plus ).
Reference: A new learning paradigm for random vector functional-link network: RVFL+. Neural Networks 122 (2020) pp.94-105
'''

from numpy.linalg import multi_dot
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV

class RVFL(object):
    """
    Parameters
    ----------
    hidden_nodes : default 50, the number of enhancement node between the input layer and the hidden layer
    random_type :default = 'uniform', please select random type from "uniform" or "gaussian"
    activation_name : default = 'sigmoid', please select an activation function from the below set: {'sigmoid', 'tanh',
                     'sin', 'hardlim', 'softlim', 'gaussianRBF', 'multiquadricRBF', 'inv_multiquadricRBF', 'tribas',
                       'inv_tribas'}
    type: default='classification', classification or regression
    Example
    --------
    model = RVFL()
    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)
    """
    def __init__(self, hidden_nodes=50, 
                 random_type="uniform", 
                 activation_name="sigmoid", 
                 type="classification"):

        self.hidden_nodes = hidden_nodes
        self.random_type = random_type
        self.activation_name = activation_name
        self.type = type
        self._activation_dict = {'sigmoid': lambda x : 1.0 / (1.0 + np.exp(-x)),
                                 "sin": lambda x: np.sin(x),
                                 "tanh": lambda x: np.tanh(x),
                                 "hardlim": lambda x:np.array(x > 0.0, dtype=float),
                                 "softlim":lambda x:np.clip(x, 0.0, 1.0),
                                 "gaussianRBF": lambda x:np.exp(-pow(x, 2.0)),
                                 "multiquadricRBF": lambda x:np.sqrt(1.0 + pow(x, 2.0)),
                                 "inv_multiquadricRBF": lambda x : 1.0 / (np.sqrt(1.0 + pow(x, 2.0))),
                                 "tribas": lambda x: np.clip(1.0 - np.fabs(x), 0.0, 1.0),
                                 "inv_tribas": lambda x: np.clip(np.fabs(x), 0.0, 1.0)}


    def _generate_randomlayer(self, X_train):
        num_samples, num_feas = X_train.shape
        # np.random.seed(0)
        if self.random_type == 'uniform':
            weights = np.random.uniform(-1.,1.,size=(num_feas, self.hidden_nodes))
            biases = np.random.uniform(-1,1.,size=(1, self.hidden_nodes))
        elif self.random_type == 'gaussian':
            weights = np.random.randn(num_feas, self.hidden_nodes)
            biases = np.random.randn(1, self.hidden_nodes)
        else:
            raise Exception('The random type is not supported now!')

        return weights, biases

    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            # print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            # print('the label must be reshaped before being transformed')
        return target

    def _softmax(self, x):
        out = np.exp(x)
        return out/ np.sum(out, axis=1, keepdims=True)

    def fit(self,train_x, train_y):

        """
        Params:
        -------
        :param train_x: a NumofSamples * NumofFeatures matrix, training data
        :param train_y: training label
        """

        self.weights, self.biases = self._generate_randomlayer(train_x)
        train_g = train_x.dot(self.weights) + self.biases
        try:
            H = self._activation_dict[self.activation_name](train_g)
        except:
            raise Exception('The activation function is not supported now!')
            
        H = np.hstack((H, train_x))
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
            
        # part_a = np.linalg.inv(H.dot(H.T))
        # part_b = one_hot_target
        # self.beta = multi_dot([H.T, part_a, part_b])
        self.beta = np.linalg.pinv(H) @ one_hot_target

    def predict(self, test_x):
        """
        Params:
        -------
        :param test_x: a NumofTestSamples * NumofFeatures matrix, test data
        :return: y_hat, the predicted labels
        """
        y_hat = self.predict_proba(test_x)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat, axis=1)
            return y_hat
        else:
            return y_hat

    def predict_proba(self, test_x):
        test_g = test_x.dot(self.weights) + self.biases
        test_h = self._activation_dict[self.activation_name](test_g)
        test_h = np.hstack((test_h, test_x))
        # print(self.beta.shape, test_h.shape)
        y_hat_temp = test_h.dot(self.beta)
        if self.type == "classification":
            y_hat_prob = self._softmax(y_hat_temp)
            # print(y_hat_temp.shape, y_hat_prob.shape)
            return y_hat_prob
        else:
            return y_hat_temp
        
    def evaluate(self, val_x, val_y, metrics=['loss', 'accuracy']):
        y_gt = to_categorical(val_y, len(set(val_y))).astype(np.float32)
        y_hat = self.predict_proba(val_x)
        y_pred = self.predict(val_x)
        # print(val_y.shape, y_gt.shape, y_hat.shape, y_pred.shape)
        
        val_loss = log_loss(y_gt, y_hat)
        val_acc = accuracy_score(val_y, y_pred)
        
        return val_loss, val_acc


class RVFLClassifier(BaseEstimator, ClassifierMixin):
    '''
    Encapsulate RVFL as a sklearn estimator
    '''
    def __init__(self, n_hidden_nodes = 20, activation = 'sigmoid'):

        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation

    def fit(self, X, y):     
        self.model = RVFL(hidden_nodes=self.n_hidden_nodes, 
                random_type="uniform", 
                activation_name=self.activation, 
                type="classification")
        self.model.fit(X, y)
        self.classes_ = np.array(list(set(y)))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class RVFLClassifierCV():

    def __init__(self, hparams = {'n_hidden_nodes': [1, 10], 'activation': ['sigmoid', 'tanh', 'sin'] }):        
        '''
        parameters  
        ----------
        hparams : hyper-parameter candidate values to be grid-searched
        '''
        self.parameters =hparams
       
    def fit(self, X, y):
        rvflc = RVFLClassifier()
        self.clf = GridSearchCV(rvflc, self.parameters, scoring = 'accuracy') # 'neg_log_loss'
        self.clf.fit(X, y)
        # print( sorted(clf.cv_results_.keys()) )
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

def create_rvfl_instance(L):
    return RVFLClassifier(L)

def create_rvflcv_instance():
    return RVFLClassifierCV()