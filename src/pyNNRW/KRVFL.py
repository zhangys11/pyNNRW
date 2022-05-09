'''
This KRVFL implementation is based on RVFL_plus ( https://github.com/pablozhang/RVFL_plus ).
Reference: A new learning paradigm for random vector functional-link network: RVFL+. Neural Networks 122 (2020) pp.94-105
'''

from numpy.linalg import multi_dot
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
                                        additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import OneHotEncoder

class KRVFL(object):
    """
    Parameters
    ----------
    kernel_name : default='rbf', the kernel function is used in KRVFL+. Please select a kernel function from the below
    set: {'rbf', 'linear', 'add_chi2', 'chi2', 'poly', 'laplace'}
    type: default='classification', classification or regression
    Example
    --------
    model = KRVFL()
    model.fit(train_x, train_y)
    y_hat = model.predict(train_x, test_x)
    """

    def __init__(self, type = 'classification', kernels = 2):

        '''
        kernels = 2 is the default case: linear + sigmoid. It is vanilla RVFL
        kernels = 1 reduces to a linear model, i.e., RVFL without hidden layer
        kernels > 2 will add more types of non-linear kernels
        '''
        self.type = type
        self.kernel_dict = {"linear": lambda x, y = None: linear_kernel(x, y),
                            "sigmoid": lambda x, y = None: sigmoid_kernel(x, y),
                            "rbf": lambda x, y = None: rbf_kernel(x, y),                            
                            "add_chi2": lambda x, y = None: additive_chi2_kernel(x, y),
                            "chi2": lambda x, y = None: chi2_kernel(x, y),
                            "poly": lambda x, y = None: polynomial_kernel(x, y),
                            "laplace": lambda x, y = None: laplacian_kernel(x, y)}
        
        self.kernel_names = list(self.kernel_dict.keys())

        # check whether kernel is in valid range
        if (kernels < 1):
            kernels = 1
        if (kernels > len(self.kernel_names ) ):
            kernels = len(self.kernel_names )

        self.kernels = kernels

    def _transform_label(self, y):
        enc = OneHotEncoder(handle_unknown='ignore')
        try:
            target = enc.fit_transform(y).toarray()
            print('the label can be transformed directly using onehotencoder')
        except:
            target = enc.fit_transform(y.reshape(-1, 1)).toarray()
            print('the label must be reshaped before being transformed')
        return target

    def _softmax(self, x):
        out = np.exp(x)
        return out/ np.sum(out, axis=1, keepdims=True)

    def fit(self, train_x, train_y):
        """
        Params:
        ---------
        :param train_x: a NumofSamples * NumofFeatures matrix, training data
        :param train_y: training label
        """
        sum_omegas = self.kernel_dict[self.kernel_names[0]](train_x) # linear kernel
        
        # add up the other non-linear kernels 
        for i in range(1, self.kernels):
            omega = self.kernel_dict[self.kernel_names[i]](train_x)
            sum_omegas = omega + sum_omegas

        # omega1, omega2 = self.kernel_dict["linear"](train_x), \
        #                 self.kernel_dict[self.kernel_name](train_x) # linear + non-linear combination
        
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
        self.beta = np.linalg.pinv(sum_omegas) @ one_hot_target

    def predict(self, train_x, test_x):
        """
        :param train_x:  a NumofSamples * NumofFeatures matrix, training data, building the kernel matrix
        :param test_x: a NumofTestSamples * NumofFeatures matrix, test data
        :return: y_hat, the predicted labels
        """
        y_hat = self.predict_proba(train_x, test_x)
        if self.type == 'classification':
            y_hat = np.argmax(y_hat, axis=1)
            return y_hat
        else:
            return y_hat

    def predict_proba(self, train_x, test_x):

        # NOTE: Unlike training, in prediction we use (test_x, train_x) as kernel input. train_x acts as anchors of the kernel operations.
        sum_omegas = self.kernel_dict[self.kernel_names[0]](test_x, train_x) # linear kernel
        
        # add up the other non-linear kernels 
        for i in range(1, self.kernels):
            omega = self.kernel_dict[self.kernel_names[i]](test_x, train_x)
            sum_omegas = omega + sum_omegas

        y_hat_temp = (sum_omegas).dot(self.beta)
        if self.type == "classification":
            y_hat_prob = self._softmax(y_hat_temp)
            return y_hat_prob
        else:
            return y_hat_temp


class KRVFLClassifier(BaseEstimator, ClassifierMixin):
    '''
    Encapsulate RVFL as a sklearn estimator
    '''
    def __init__(self, kernels = 5):

        self.model = KRVFL(kerenes = kernels, type="classification")

    def fit(self, X, y):     

        self.model.fit(X, y)
        self.classes_ = np.array(list(set(y)))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)    


class KRVFLClassifierCV():

    def __init__(self, hparams = {'kernels': [1, 10] }):        
        '''
        parameters  
        ----------
        hparams : hyper-parameter candidate values to be grid-searched
        '''
        self.parameters =hparams
       
    def fit(self, X, y):
        rvflc = KRVFLClassifier()
        self.clf = GridSearchCV(rvflc, self.parameters, scoring = 'accuracy') # 'neg_log_loss'
        self.clf.fit(X, y)
        # print( sorted(clf.cv_results_.keys()) )
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

def create_krvfl_instance(L): # L is the non-linear kernel number
    return KRVFLClassifier(L)

def create_rvflcv_instance():
    return KRVFLClassifierCV()