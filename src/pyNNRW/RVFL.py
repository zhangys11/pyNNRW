import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from . import to_categorical


class RVFL:
    """
    A simple RVFL classifier or regression.
    This implementation is based on https://github.com/Xuyang-Huang/Simple-RVFL-python/tree/main

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter. Controls the direct-link strength.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """
    def __init__(self, n_nodes, lam, w_random_vec_range = [-1, 1],
                 b_random_vec_range = [0, 1], activation = 'sigmoid',
                 same_feature=False,
                 task_type='classification'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam # controls the direct-link strength
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        # self.data_std = None
        # self.data_mean = None
        self.same_feature = same_feature
        self.task_type = task_type

    def train(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        # data = self.standardize(data)  # Normalization data
        n_sample = len(data) #样本数量
        n_feature = len(data[0]) #特征数量
        self.random_weights = self.get_random_vectors(n_feature, self.n_nodes, self.w_random_range) #构成随机权重值
        self.random_bias = self.get_random_vectors(1, self.n_nodes, self.b_random_range) #构成随机偏置值
        
        #点积，并添加偏置项，使用激活函数处理，得到激活值h
        h = self.activation_function(np.dot(data, self.random_weights) + np.dot(np.ones([n_sample, 1]), self.random_bias))
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        if self.task_type == 'classification':
            y = self.one_hot(label, n_class) #进行独热编码
        else:
            y = label
        if n_sample > (self.n_nodes + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                 When regression, return the output of rvfl.
        """
        # data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == 'regression':
            return output

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        # data = self.standardize(data)  # Normalization data
        h = self.activation_function(np.dot(data, self.random_weights) + self.random_bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            result = np.argmax(output, axis=1)
            acc = np.sum(np.equal(result, label)) / len(label)
            return acc
        elif self.task_type == 'regression':
            mae = np.mean(np.abs(output - label))
            return mae

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x):
        if self.same_feature is True:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

class RVFLClassifier(BaseEstimator, ClassifierMixin):
    '''
    Encapsulate RVFL as a sklearn estimator
    '''
    def __init__(self, n_hidden_nodes = 20, activation = 'sigmoid'):

        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        self.model = RVFL(n_nodes=self.n_hidden_nodes, lam=0.1,
                          activation=self.activation, task_type='classification')
        # print(self.n_hidden_nodes, self.activation)
        self.classes_ = 0
        self.n_features_in_ = 0

    def fit(self, X, y):        
        self.model.train(X, y, len(set(y)))
        self.classes_ = np.unique(y) # self.classes_ = np.array(list(set(y)))

        '''
        n_features_in_ is the number of features that an estimator expects.
        In most cases, the n_features_in_ attribute exists only once fit has been called, but there are exceptions.
        '''
        self.n_features_in_ = X.shape[1]

    def predict(self, X):
        predition, _ = self.model.predict(X)
        return predition

    def predict_proba(self, X):
        _, proba = self.model.predict(X)
        return proba

    def evaluate(self, X, y):
        '''
        Return classification accuracy by default
        '''
        return self.model.eval(X, y)


class RVFL_v2(object):
    """
    This RVFL implementation is based on RVFL_plus ( https://github.com/pablozhang/RVFL_plus ).
    Reference: A new learning paradigm for random vector functional-link network: RVFL+. Neural Networks 122 (2020) pp.94-105

    Parameters
    ----------
    hidden_nodes : default 50, the number of enhancement node between the input layer and the hidden layer
    random_type :default = 'uniform', please select random type from "uniform" or "gaussian"
    activation_name : default = 'sigmoid', please select an activation function from the below set: {'sigmoid', 'tanh',
                     'sine', 'hardlim', 'softlim', 'gaussianRBF', 'multiquadricRBF', 'inv_multiquadricRBF', 'tribas',
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
                                 "sine": lambda x: np.sin(x),
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
        
    def evaluate(self, val_x, val_y, metrics=['loss', 'accuracy', 'precision', 'recall']):
        '''        
        Returns
        -------
        'loss', 'accuracy', 'precision', 'recall'
        '''

        y_gt = to_categorical(val_y, len(set(val_y))).astype(np.float32)
        y_hat = self.predict_proba(val_x)
        y_hat = np.nan_to_num(y_hat) # replace nan
        y_pred = self.predict(val_x)
        # print(val_y.shape, y_gt.shape, y_hat.shape, y_pred.shape)

        val_loss = log_loss(y_gt, y_hat)
        val_acc = accuracy_score(val_y, y_pred)
        val_precision = precision_score(val_y, y_pred) 
        val_recall = recall_score(val_y, y_pred)
        
        return val_loss, val_acc, val_precision, val_recall


class RVFLClassifier_v2(BaseEstimator, ClassifierMixin):
    '''
    Encapsulate RVFL as a sklearn estimator
    '''
    def __init__(self, n_hidden_nodes = 20, activation = 'sigmoid'):

        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation

    def fit(self, X, y):
        self.model = RVFL_v2(hidden_nodes=self.n_hidden_nodes,
                random_type="gaussian",
                activation_name=self.activation,
                type="classification")
        self.model.fit(X, y)
        self.classes_ = np.unique(y) # self.classes_ = np.array(list(set(y)))

        '''
        n_features_in_ is the number of features that an estimator expects.
        In most cases, the n_features_in_ attribute exists only once fit has been called, but there are exceptions.
        '''
        self.n_features_in_ = X.shape[1]

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        '''
        Return the predicted probability distribution.
        '''
        return self.model.predict_proba(X)

    def evaluate(self, X, y, metrics=['loss', 'accuracy', 'precision', 'recall']):
        '''
        Return loss, accuracy, precision, recall by default
        '''
        return self.model.evaluate(X, y, metrics=metrics)

class RVFLClassifierCV():
    '''
    Encapsulate RVFL as a sklearn estimator and use CV to optimize its hyper-parameters
    '''

    def __init__(self, hparams = {'n_hidden_nodes': [1, 10], 'activation': ['sigmoid', 'tanh', 'sine'] }):      
        '''
        parameters  
        ----------
        hparams : hyper-parameter candidate values to be grid-searched
        '''
        self.parameters =hparams
        self.clf = None
        self.classes_ = []
       
    def fit(self, X, y):
        rvflc = RVFLClassifier()
        self.clf = GridSearchCV(rvflc, self.parameters, scoring = 'accuracy') # 'neg_log_loss'
        self.clf.fit(X, y)
        # print( sorted(clf.cv_results_.keys()) )
        self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

def create_rvfl_instance(L, activation = 'sigmoid', flavor = 'v1'):
    '''
    Return a sklearn estimator compatible RVFL instance

    Parameters
    ----------
    L : hidden layer nodes
    activation : activation function
    flavor : 'v1' or 'v2'. Choose either of the two RVFL implementations. 
    '''
    if flavor == 'v1':
        return RVFLClassifier(L, activation)
    else:
        return RVFLClassifier_v2(L, activation)

def create_rvflcv_instance():
    '''
    Return a sklearn estimator compatible RVFL CV instance
    '''
    return RVFLClassifierCV()