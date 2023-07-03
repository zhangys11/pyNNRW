from inspect import stack
from numpy.linalg import multi_dot
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
                                        additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import OneHotEncoder

class KNNRW(object):
    """
    kernel-NNRW
    """

    def __init__(self, N = 1, 
    type = 'classification', 
    kernels = ['linear','sigmoid'], 
    flavor = 'stack' 
    ):

        '''
        Parameters
        ----------
        N : int. Some kernels' params need n_features to get the default value.

        kernels : int or array. If kernels is an integer n, will use the first n built-in kernels. 
                  kernels can also be an array of kernel names (duplicate is allowed)

        kernels = ['linear','sigmoid'] is the default case: linear + sigmoid. It is kernel version of vanilla RVFL
        kernels = ['linear'] reduces to a linear model
        kernels = ['sigmoid'] reduceds to k-ELM.
        kernels > 2 will add more types of non-linear kernels

        flavor: 'stack' , 'sum', or 'stack+sum'.

        '''
        self.type = type
        self.flavor = flavor

        # the candidate kernels are pre-selected by Raman domain datasets
        self.kernel_dict = {"linear": lambda x, y = None: linear_kernel(x, y),
                            "sigmoid_0.1": lambda x, y = None: sigmoid_kernel(x, y, 0.1/N),
                            "sigmoid": lambda x, y = None: sigmoid_kernel(x, y, 1.0/N),
                            "sigmoid_10": lambda x, y = None: sigmoid_kernel(x, y, 10.0/N),
                            "sigmoid_100": lambda x, y = None: sigmoid_kernel(x, y, 100.0/N),
                            "rbf_0.1": lambda x, y = None: rbf_kernel(x, y, 0.1/N),
                            "rbf_0.3": lambda x, y = None: rbf_kernel(x, y, 0.3/N),
                            "rbf": lambda x, y = None: rbf_kernel(x, y),
                            "rbf_3": lambda x, y = None: rbf_kernel(x, y, 3.0/N),
                            "rbf_10": lambda x, y = None: rbf_kernel(x, y, 10.0/N),    
                            "laplace_0.5": lambda x, y = None: laplacian_kernel(x, y, 0.5/N),
                            "laplace": lambda x, y = None: laplacian_kernel(x, y),
                            "laplace_2": lambda x, y = None: laplacian_kernel(x, y, 2.0/N),
                            "laplace_4": lambda x, y = None: laplacian_kernel(x, y, 4.0/N),
                            "cosine": lambda x, y = None: cosine_similarity(x, y),                            
                            "chi2_0.001": lambda x, y = None: chi2_kernel(x, y, 0.001),
                            "chi2_0.01": lambda x, y = None: chi2_kernel(x, y, 0.01),
                            "chi2_0.1": lambda x, y = None: chi2_kernel(x, y, 0.1),
                            "chi2": lambda x, y = None: chi2_kernel(x, y),
                            "add_chi2": lambda x, y = None: additive_chi2_kernel(x, y),
                            "poly": lambda x, y = None: polynomial_kernel(x, y, 0.001/N)
                            }    

        self.kernel_names = list(self.kernel_dict.keys())

        if isinstance(kernels,int): # type(kernels) == 'int': # int
            # check whether kernel is in valid range
            if (kernels < 1):
                kernels = 1
            if (kernels > len(self.kernel_names ) ):
                kernels = len(self.kernel_names )

            self.kernels = []
            for i in range(kernels):
                self.kernels.append(self.kernel_names[i])

        else: # array
            self.kernels = kernels

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

    def fit(self, train_x, train_y, verbose = 0):
        """
        Params:
        ---------
        :param train_x: a NumofSamples * NumofFeatures matrix, training data
        :param train_y: training label
        """

        n_features = train_x.shape[1]

        summed_omegas = self.kernel_dict[self.kernels[0]](train_x) # linear kernel
        stacked_omega = self.kernel_dict[self.kernels[0]](train_x)

        # add up the other non-linear kernels 
        for i in range(1, len(self.kernels) ): 
            omega = self.kernel_dict[self.kernels[i]](train_x)
            summed_omegas = omega + summed_omegas
            stacked_omega = np.hstack((stacked_omega,omega))

        # omega1, omega2 = self.kernel_dict["linear"](train_x), \
        #                 self.kernel_dict[self.kernel_name](train_x) # linear + non-linear combination
        
        if self.type == 'classification':
            one_hot_target = self._transform_label(train_y)
        elif self.type == 'regression':
            one_hot_target = train_y
        else:
            raise Exception("The type is not supported now! please select classification or regression.")
        
        if self.flavor == 'stack':
            OMEGA = stacked_omega
        elif self.flavor == 'sum':
            OMEGA = summed_omegas
        else: # stack + sum
            OMEGA = np.hstack((stacked_omega, summed_omegas))

        self.beta = np.linalg.pinv(OMEGA) @ one_hot_target
        if verbose:
            print('OMEGA shape: ', OMEGA.shape, ', BETA shape: ', self.beta.shape)

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
        summed_omegas = self.kernel_dict[self.kernels[0]](test_x, train_x) # linear kernel
        stacked_omega = self.kernel_dict[self.kernels[0]](test_x, train_x)

        # add up the other non-linear kernels 
        for i in range(1, len(self.kernels)) :
            omega = self.kernel_dict[self.kernels[i]](test_x, train_x)
            summed_omegas = omega + summed_omegas
            stacked_omega = np.hstack((stacked_omega,omega))

        if self.flavor == 'stack':
            OMEGA = stacked_omega
        elif self.flavor == 'sum':
            OMEGA = summed_omegas
        else: # stack + sum
            OMEGA = np.hstack((stacked_omega, summed_omegas))

        y_hat_temp = (OMEGA).dot(self.beta)
        if self.type == "classification":
            y_hat_prob = self._softmax(y_hat_temp)
            return y_hat_prob
        else:
            return y_hat_temp

def KernelTuning(X):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    '''

    # linear_kernel返回Gram Matrix: n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的格拉姆矩阵(Gram matrix)
    plt.imshow(linear_kernel(X,X)) # return the Gram matrix, i.e. X @ Y.T.
    plt.axis('off')
    plt.title('linear kernel (Gram matrix). \nNo tunable params.')
    plt.show()


    for gamma in [0.1 /X.shape[1] , 0.33 /X.shape[1] , 
        1 /X.shape[1], 3.33 /X.shape[1], 10/X.shape[1]]:
    
        plt.imshow(rbf_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('rbf kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||^2) \ndefault gamma = 1/n.')
        plt.show()


    for gamma in [0.5 /X.shape[1] , 1 /X.shape[1], 2 /X.shape[1], 4/X.shape[1]]:

        plt.imshow(laplacian_kernel(X,X, gamma = gamma)) 
        plt.axis('off')
        plt.title('laplacian kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||_1) \ndefault gamma = 1/n.')
        plt.show()
    

    for gamma in [0.1 /X.shape[1] , 1 /X.shape[1], 10/X.shape[1], 100/X.shape[1]]:
    
        plt.imshow(sigmoid_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('sigmoid kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = tanh(gamma <X, Y> + coef0) \ndefault gamma = 1/n.')
        plt.show()
        

    for gamma in [0.001 /X.shape[1] , 0.01 /X.shape[1] , 0.1 /X.shape[1], 1 /X.shape[1]]:
 
        plt.imshow(polynomial_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('polynomial kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = (gamma <X, Y> + coef0)^degree \ndefault gamma = 1/n, degree = 3')
        plt.show()


    for gamma in [0.001, 0.01 , 0.1 , 1]:
   
        # chi2 requires non-negative input
        plt.imshow(chi2_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('chi2 kernel (gamma = ' +str(gamma)+ '). \nk(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)]) \ndefault gamma = 1')
        plt.show()


    # additive chi2 requires non-negative input
    plt.imshow(additive_chi2_kernel(X, X))
    plt.axis('off')
    plt.title('additive chi2 kernel. \nk(x, y) = -Sum [(x - y)^2 / (x + y)] \nNo tunable params.')
    plt.show()


    plt.imshow(cosine_similarity(X,X))
    plt.axis('off')
    plt.title('cosine kernel. \nK(X, Y) = <X, Y> / (||X||*||Y||) \nNo tunable params.')
    plt.show()

    return

def KNNRWClassifierGridSearch(x_train, x_test, t_train, t_test, K=22, verbose = 0):
    '''
    Perform a grid search for the best kernel combinations.

    Parameters
    ----------
    K : int, the maximum kernel numbers. default = 22
    '''

    N = x_train.shape[1]

    taccs = []
    vaccs = []
        
    # Perform a grid search
    for flavor in ['sum','stack','stack+sum']:
        
        tacc = []
        vacc = []
        kernels = []
        
        for k in list(range(1, K)):
            
            # ===============================
            # Instantiate
            # ===============================
            clf = KNNRWClassifier(N = N, kernels = k, flavor = flavor)

            if verbose:
                print('===== Flavor: ' + flavor + ', kernel = ' + str(clf.model.kernels) + ' =====')
            kernels.append( str(clf.model.kernels) )

            # ===============================
            # Training
            # ===============================
            clf.fit(x_train, t_train, verbose = verbose)
            t_pred = clf.predict(x_train, x_train)
            ACC = np.mean(t_train == t_pred)
            if verbose:
                print("ACC on training set = ", ACC)
            tacc.append(ACC)

            # ===============================
            # Validation
            # ===============================
            t_pred = clf.predict(x_train, x_test)
            ACC = np.mean(t_test == t_pred)
            if verbose:
                print("ACC on test set = ", ACC)
            vacc.append(ACC)
            
            if verbose:
                print('========= \n\n')
            
        taccs.append(tacc)
        vaccs.append(vacc)

    for i, flavor in enumerate(['sum','stack','stack+sum']):
        plt.figure(figsize = (K, 4))
        plt.title(flavor)
        plt.scatter(kernels, taccs[i], label = 'train acc')
        plt.scatter(kernels, vaccs[i], label = 'val acc')
        plt.legend()
        plt.xticks(rotation=-90)
        plt.show()

    return taccs, vaccs


class KNNRWClassifier(BaseEstimator, ClassifierMixin):
    '''
    Encapsulate as a sklearn estimator
    '''
    def __init__(self, N = 1, kernels = 5, flavor = 'stack'):
        self.model = KNNRW(N = N, kernels = kernels, type="classification", flavor=flavor) 

    def fit(self, X, y, verbose = 0):     
        
        self.model.fit(X, y, verbose = verbose)
        # self.classes_ = np.array(list(set(y)))

    def predict(self, X_train, X_test):
        return self.model.predict(X_train, X_test)

    def predict_proba(self, X_train, X_test):
        return self.model.predict_proba(X_train, X_test)    

def create_knnrw_instance(L): # L is the non-linear kernel number
    return KNNRWClassifier(L)