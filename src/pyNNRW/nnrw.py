import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import StackingClassifier #,HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import r2_score

from . import to_categorical

warnings.filterwarnings("ignore")

#region Performance Test. Compare NNRW with mainstream classifier models.

def ELMClf(X, y, L = 100, verbose = False):
    '''
    A wrapper for ELM.
    L represents the number of hidden layer nodes.
    '''

    # ===============================
    # Load dataset
    # ===============================
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
    t_train = to_categorical(t_train, n_classes).astype(np.float32)
    t_test = to_categorical(t_test, n_classes).astype(np.float32)
    # print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

    # ===============================
    # ELM parameters
    # ===============================
    n_hidden_nodes = L #x_train.shape[1]
    loss = 'mean_squared_error' # 'mean_absolute_error'
    activation = 'sigmoid' # 'identity'

    # ===============================
    # Instantiate ELM
    # ===============================
    model = ELM(
        n_input_nodes = x_train.shape[1],
        n_hidden_nodes = n_hidden_nodes,
        n_output_nodes = n_classes,
        loss = loss,
        activation=activation,
        name='elm'
    )

    # ===============================
    # Training
    # ===============================
    time1 = time.time_ns()        
    model.fit(x_train, t_train)
    time2 = time.time_ns()
    # print('ELM took {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    train_loss, train_acc, train_precision, train_recall = model.evaluate(x_train, t_train, 
    metrics=['loss', 'accuracy', 'precision', 'recall'])
    if (verbose):
        print('train_loss: %f' % train_loss) # loss value
        print('train_acc: %f' % train_acc) # accuracy
        print('train_precision: %f' % train_precision)
        print('train_recall: %f' % train_recall) # uar (unweighted average recall)


    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc, val_precision, val_recall = model.evaluate(x_test, t_test, 
    metrics=['loss', 'accuracy', 'precision', 'recall'])
    if (verbose):
        print('val_loss: %f' % val_loss)
        print('val_acc: %f' % val_acc)
        print('val_precision: %f' % val_precision)
        print('val_recall: %f' % val_recall)
        print('train/fit time: {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    return train_acc, val_acc, (time2-time1)/ (10 ** 6)


def RVFLClf(X, y, L = 100, verbose = False):
    '''
    A wrapper for RVFL.
    L represents the number of hidden layer nodes.
    '''

    # ===============================
    # Load dataset
    # ===============================
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
    
    ##  NOTE: Due to the inner implementation, RVFL doesn't need to do this one-hot encoding.
    # t_train = to_categorical(t_train, n_classes).astype(np.float32)
    # t_test = to_categorical(t_test, n_classes).astype(np.float32)
    
    # ===============================
    # RVFL parameters
    # ===============================
    n_hidden_nodes = L #x_train.shape[1]
    loss = 'mean_squared_error' # 'mean_absolute_error'
    activation = 'sigmoid' # 'identity'

    # ===============================
    # Instantiate RVFL
    # ===============================
    model = RVFL(
        n_hidden_nodes,
        random_type="uniform",
        type="classification"
    )

    # ===============================
    # Training
    # ===============================
    time1 = time.time_ns()        
    model.fit(x_train, t_train)
    time2 = time.time_ns()
    # print('RVFL took {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
        
    train_loss, train_acc, train_precision, train_recall = model.evaluate(x_train, t_train)
    if (verbose):
        print('train_loss: %f' % train_loss) # loss value
        print('train_acc: %f' % train_acc) # accuracy
        print('train_precision: %f' % train_precision)
        print('train_recall: %f' % train_recall) # uar (unweighted average recall)


    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc, val_precision, val_recall = model.evaluate(x_test, t_test)
    if (verbose):
        print('val_loss: %f' % val_loss)
        print('val_acc: %f' % val_acc)
        print('val_precision: %f' % val_precision)
        print('val_recall: %f' % val_recall)
        print('train/fit time: {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    return train_acc, val_acc, (time2-time1)/ (10 ** 6)


def MLPClf(X, y, L = 100, verbose = False, alpha = 0.1, iters = 150):
    '''
    A wrapper for MLP, which implements a GD (gradient-descent) training.
    L represents the number of hidden layer nodes.
    '''

    # ===============================
    # Load dataset
    # ===============================
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
    t_train = to_categorical(t_train, n_classes).astype(np.float32)
    t_test = to_categorical(t_test, n_classes).astype(np.float32)
    
    # ===============================
    # Instantiate MLP
    # ===============================
    network = MLP(input_size=X.shape[1], hidden_size=L, output_size = n_classes)

    # ===============================
    # Training - MBGD
    # ===============================
    time1 = time.time_ns()       
    
    train_size = x_train.shape[0]
    batch_size = round(train_size / 2) 
    learning_rate = alpha

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters): # 学习率=0.1时，100是最小可接受的值，低于此值，acc会变低，并出现较大波动
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.bp_gradient(x_batch, t_batch) # BP is much faster than the numerical gradient method

        # update parameters
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

    time2 = time.time_ns()
        
    train_acc = network.accuracy(x_train, t_train)
    if (verbose):
        print('train_acc: %f' % train_acc) # accuracy

    # ===============================
    # Validation
    # ===============================
    val_acc = network.accuracy(x_test, t_test)
    if (verbose):
        print('val_acc: %f' % val_acc)
        print('train/fit time: {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    return train_acc, val_acc, (time2-time1)/ (10 ** 6)


def SVMClf(X, y, gamma = 1, verbose = False):
    '''
    A wrapper for SVM. Internally, we use the one-vs-one strategy for multi-class.
    gamma represents the gamma hyper-parameter of the rbf kernel, i.e., 1/ (2*sigma^2).
    '''

    # ===============================
    # Load dataset
    # ===============================
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
    #t_train = to_categorical(t_train, n_classes).astype(np.float32)
    #t_test = to_categorical(t_test, n_classes).astype(np.float32)
    # print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

    # ===============================
    # Instantiate SVC
    # ===============================
    
    model = OneVsOneClassifier( SVC(kernel='rbf', gamma = gamma, C = 2) ) # C = 2 is selected by CV

    # ===============================
    # Training
    # ===============================
    time1 = time.time_ns()        
    model.fit(x_train, t_train)
    time2 = time.time_ns()
    # print('ELM took {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
        
        
    t_pred = model.predict(x_train)
    train_acc = np.mean(t_train == t_pred)

    # train_loss, train_acc, train_precision, train_recall = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy', 'precision', 'recall'])
    if (verbose):
        print('train_acc: %f' % train_acc) # accuracy
        #print('train_precision: %f' % train_precision)
        #print('train_recall: %f' % train_recall) # uar (unweighted average recall)


    # ===============================
    # Validation
    # ===============================
    t_pred = model.predict(x_test)
    val_acc = np.mean(t_test == t_pred)
    
    # val_loss, val_acc, val_precision, val_recall = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy','precision', 'recall'])
    if (verbose):
        print('val_acc: %f' % val_acc)
        #print('val_precision: %f' % val_precision)
        #print('val_recall: %f' % val_recall)
        print('train/fit time: {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    return train_acc, val_acc, (time2-time1)/ (10 ** 6)


def TreeClf(X, y, max_depth = 1, verbose = False):
    '''
    A wrapper for DTC (decision tree classifier). Internally, we use the one-vs-one strategy for multi-class.
    max_depth represents the maximum tree depth.
    '''

    # ===============================
    # Load dataset
    # ===============================
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)

    # ===============================
    # Instantiate DTC
    # ===============================
    
    model =  DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)  # A 1-deep tree is also called a decision stump

    # ===============================
    # Training
    # ===============================
    time1 = time.time_ns()        
    model.fit(x_train, t_train)
    time2 = time.time_ns()
    # print('ELM took {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
        
        
    t_pred = model.predict(x_train)
    train_acc = np.mean(t_train == t_pred)

    # train_loss, train_acc, train_precision, train_recall = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy', 'precision', 'recall'])
    if (verbose):
        print('train_acc: %f' % train_acc) # accuracy
        #print('train_precision: %f' % train_precision)
        #print('train_recall: %f' % train_recall) # uar (unweighted average recall)


    # ===============================
    # Validation
    # ===============================
    t_pred = model.predict(x_test)
    val_acc = np.mean(t_test == t_pred)
    
    # val_loss, val_acc, val_precision, val_recall = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy','precision', 'recall'])
    if (verbose):
        print('val_acc: %f' % val_acc)
        #print('val_precision: %f' % val_precision)
        #print('val_recall: %f' % val_recall)
        print('train/fit time: {:.3f} ms'.format((time2-time1)/ (10 ** 6)))
    
    return train_acc, val_acc, (time2-time1)/ (10 ** 6)


def PerformenceTest(func, X, y, Ls = list(range(1, 80)), plot = False):
    '''
    Run one round performence test. 

    Parameters
    ----------
    func : a funciton callback that encapsulates a specific learning model.
    X,y : the dataset for supervised learning
    Ls: a list of hyper-parameter candidate values to test. The specific meaning of the hyper-parameter is determined by the func.
    plot: whether to display the generated curves.

    Returns
    -------
    Tacc : the training accuracies
    Vacc : the validation accuracies
    T : the training/fitting time
    '''

    Tacc = []
    Vacc = []
    T = []

    for L in Ls:
        ta, va, t = func(X, y, L)
        Tacc.append(ta)
        Vacc.append(va)
        T.append(t)

    if plot:        
        plt.figure(figsize = (12,4))
        plt.plot(Ls, Tacc, label="train acc")
        plt.plot(Ls, Vacc, label = "val acc")
        plt.legend()
        plt.show()

        plt.figure(figsize = (12,4))
        plt.plot(Ls, T, label="train/fit time (ms)")
        plt.legend()
        plt.show()
    
    return Tacc, Vacc, T


def PerformenceTests(func, X, y, Ls = list(range(1, 80)), N = 20):

    '''
    Run multiple rounds of performence test to get the average.

    Parameters
    ----------
    func : a funciton callback that encapsulates a specific learning model.
    X,y : the dataset for supervised learning
    Ls : a list of hyper-parameter candidate values to test. The specific meaning of the hyper-parameter is determined by the func.
    plot : whether to display the generated curves.
    N : rounds of test

    Returns
    -------
    MTacc : the averaged training accuracies
    MVacc : the averaged validation accuracies
    MT : the averaged training/fitting time
    '''


    Taccs = []
    Vaccs = []
    Ts = []
    
    for i in range(0, N):
        Tacc, Vacc, T = PerformenceTest(func, X, y, Ls = Ls, plot = False)
        Taccs.append(Tacc)
        Vaccs.append(Vacc)
        Ts.append(T)
        
    MTacc = np.array(Taccs).mean(axis = 0)
    MVacc = np.array(Vaccs).mean(axis = 0)
    MT = np.array(Ts).mean(axis = 0)
    
    plt.figure(figsize = (12,4))
    plt.plot(Ls, MTacc, label="train acc")
    plt.plot(Ls, MVacc, label = "val acc")
    plt.legend()
    plt.show()

    plt.figure(figsize = (12,4))
    plt.plot(Ls, MT, label="train/fit time (ms)")
    plt.legend()
    plt.show()
    
    return MTacc, MVacc, MT

#endregion

#region ensembles

class LogisticRegressionX(LogisticRegression, ClassifierMixin):
    '''
    An extended version of logistic regression that handles NaN.
    '''
    
    def __init__(
        self,
        penalty="l2",
        tol=1e-4,
        C=1.0,
        l1_ratio=None):
        
        LogisticRegression.__init__(self, penalty = penalty, tol=tol, C=C, l1_ratio=l1_ratio)
        
    def fit(self, X, y):

        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        clf = LogisticRegression.fit(self, X=X, y=y)
        clf.coef_ = np.nan_to_num(clf.coef_)
        clf.intercept_ = np.nan_to_num(clf.intercept_)
        return clf

    def predict_proba(self, X):
        X = np.nan_to_num(X)       
        yh = LogisticRegression.predict_proba(self, X=X)
        yh = np.nan_to_num(yh)
        return yh

    def predict(self, X):
        X = np.nan_to_num(X)
        yp = LogisticRegression.predict(self, X=X)
        yp = np.nan_to_num(yp)
        return yp
    
    def score(self, X, y):
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return LogisticRegression.score(X=X, y=y)   

def homo_stacking(X, y, create_base_estimator, 
                  meta_learner = LogisticRegressionX,
                  Ns = [1, 2, 5], Ls = [1, 2, 5, 10, 20], 
                  test_size = .3,
                  random_state = None,
                  WITH_CONTEXT = False,
                  xlabel = '',
                  YLIM = None):
    '''
    create_base_estimator # a function that create base learner instanes
    Ns = [1, 2, 5] # number of base estimators
    Ls = [1, 2, 5, 10, 20] # base learner's specific hyper-parameter    
    '''

    # pbar = tqdm(total = repeat * len(Ns) * len(Ls), position=0, leave=True) # stay on top

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state,
    )
        
    dic_acc = {}
    repeat = 1 # no need to repeat, each run is now deterministic.

    plt.figure(figsize = (14, 6))

    for N in Ns:
        ACCs = []

        for L in Ls:

            accs = []

            for _ in range(repeat):

                estimators = []
                for i in range(N):
                    estimators.append((str(i), create_base_estimator(L))) # relu is better than logistic

                clf = StackingClassifier(
                    estimators=estimators, final_estimator=meta_learner(), passthrough = WITH_CONTEXT
                )

                # acc = acc + clf.fit(X_train, y_train).score(X_test, y_test)
                accs.append(clf.fit(X_train, y_train).score(X_test, y_test))

                # pbar.update()
            
            # remove the biggest and smallest from accs and get mean 
            # print('acc of all runs:', np.round(accs,2))
            averaged_acc = sum(accs)/repeat
            if repeat >= 3:
                averaged_acc = sum(sorted(accs)[1:-1]) / (len(accs) - 2)
            
            ACCs.append(averaged_acc)
            dic_acc[(N, L)] = averaged_acc

        plt.plot(Ls, ACCs, '--', label = 'N='+str(N), marker='o') # fillstyle='none'
        # plt.scatter(Ls, ACCs, s = 50)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1)) # only show integer
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(.1)) # only show integer
    
    Ls = np.array(Ls)
    plt.xlim( (Ls.min() - .5, Ls.max() + .5) )
    plt.legend()
    plt.xlabel('Hyper-parameter' + xlabel)
    plt.ylabel("Classfication Accuracy")
    if YLIM is None:
        plt.ylim( (np.min(list(dic_acc.values())) - .05, 1.05) )
    else:
        plt.ylim(YLIM) # e.g., (.4, 1.05)
    plt.show()

    print('best test accuracy: ', max(dic_acc.values()), 'N, L = ', [key for key, value in dic_acc.items() if value == max(dic_acc.values())])

    return dic_acc
    
def hetero_stacking(X, y, estimators,
                    test_size = .3, random_state = None,
                    WITH_CONTEXT = False, repeat = 5):
    '''
    estimators: a batch of base learners. Each learner should be derived from the BaseEstimator type.    
    '''
    acc = 0
    repeat = 1
    
    for _ in range(repeat):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y,
            test_size= test_size,
            random_state=random_state
        )

        clf = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegressionCV(),
            passthrough = WITH_CONTEXT
        )

        acc = acc + clf.fit(X_train, y_train).score(X_test, y_test)
    
    return acc / repeat

class FSSE(BaseEstimator, ClassifierMixin):
    '''
    Feature subspace based ensemble (FSSE)
    
    parameters
    ----------
    create_base_estimator_cv : { create_elmcv_instance, create_rvflcv_instance }. 
        a function callback that creates a cv optimized base estimator. 
    feature_split : an integer or 'all'. The interval or "window size" for each split. 
    meta_l1_ratios : l1_ratios to be tried for the meta-learner. We recommend > 0.5 values to achieve sparsity

    Sample Code
    -----------
    fsse = FSSE(create_elmcv_instance, feature_split = split)
    fsse.fit(X, y)
    acc = fsse.evaluate(X, y) # accuracy
    
    '''
    def __init__(self, create_base_estimator_cv, feature_split = 'all', meta_l1_ratios = [0.5,0.6,0.7,0.8,0.9,1.0]):

        self.create_base_estimator_cv = create_base_estimator_cv
        self.feature_split = feature_split
        self.meta_l1_ratios = meta_l1_ratios
        self.base_learners = {}
        self.meta_learner = None

    def fit(self, X, y):
        
        if isinstance(self.feature_split, int):
            self.feature_split = abs(self.feature_split)

        if (self.feature_split == 'all' or self.feature_split >= X.shape[1]):
            self.feature_split = X.shape[1] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
                
        predicts = []
        for i in range(0, X.shape[1], self.feature_split): # TODO: padding to N*feature_split, or special treatment for the last group.
            clfcv = self.create_base_estimator_cv().fit(X_train[:,i:i+ self.feature_split], y_train)
            self.base_learners[(i, i + self.feature_split)] = clfcv.clf.best_estimator_ # the inner best model
            predicts.append( clfcv.predict(X_test[:,i:i+ self.feature_split]) )

        predicts = np.array(predicts).T
        # print(predicts.shape)
        # print(X_test.shape, y_test.shape)
        self.meta_learner = LogisticRegressionCV(penalty = 'elasticnet',  # use elasticnet penalty to get sparse result
                                            l1_ratios = self.meta_l1_ratios, # require at least 0.5 L1 ratio for sparsity
                                           solver = 'saga' # only saga support elasticnet penalty
                                           ).fit(predicts ,y_test)
        # print(meta_learner.l1_ratio_)
        # plt.plot(meta_learner.coef_[0])
        # return base_learners, meta_learner  

    def base_predict(self, X):
        '''
        base leaners' prediction, used as the input for meta-learner
        '''
        predicts = []
        for k, b in self.base_learners.items():
            FS = X[:,k[0]:k[1]]
            yhat = b.predict(FS)
            predicts.append(yhat)

        return np.array(predicts).T

    def predict(self, X):

        predicts = self.base_predict(X)
        return self.meta_learner.predict(predicts)
    
    def predict_proba(self, X):

        predicts = self.base_predict(X)
        return self.meta_learner.predict_proba(predicts)
    
    def evaluate(self, X, y, metrics=['accuracy','r2']):
        '''
        Parameters
        ----------
        metrics : an array
            accuracy - Classification Accuracy
            r2 - The coefficient of determination. 
                R2 is a number between 0 and 1 that measures how well a statistical model 
                predicts an outcome.
        '''
        y_hat = self.predict(X)
        acc = (y_hat == y).mean()

        if len(set(y)) == 2:
            y_proba = np.array(self.predict_proba(X))
            y_reg = y_proba[:,1] # suppose col 1 is for Y = 1
            r2a = r2_score(y, y_reg) # an alternative r2 

        r2 = r2_score(y, y_hat) # R2 is a regression score, we use the predicted probs as a regression for y label.

        return acc, r2

    # def plot_feature_importance():
    #    plot_feature_importance(np.abs(self.meta_learner.coef_[0]), 'FSSE FS Result') # need to include feature_importance.py

    def get_important_features(self):
        '''
        Sample Code
        -----------
        biggest_fsse_fs, fs_importance = fsse.get_important_features()
        xfsse = X_scaled[:,biggest_fsse_fs] # 前N个系数 non-zero
        plot_feature_importance(np.abs(fs_importance), 'FSSE FS Result') # require feature_importance.py

        return
        ------
        biggest_fsse_fs : the most important feature subset selected by meta-learner's coef
        fs_importance : the most important features' importance
        '''

        # print(fsse.meta_learner.l1_ratio_)
        N = np.count_nonzero(self.meta_learner.coef_[0])
        
        biggest_fsse_gs = (np.argsort(np.abs(self.meta_learner.coef_[0]))[-N:])[::-1] # take last N item indices and reverse (ord desc)
        biggest_fsse_fs = []
        ranges = list(self.base_learners)

        for idx in biggest_fsse_gs:
            biggest_fsse_fs = biggest_fsse_fs + list(  range(ranges[idx][0], ranges[idx][1]) ) # list(range(idx*self.feature_split, (idx+1)*self.feature_split))
    
        biggest_fsse_fs = np.array(biggest_fsse_fs)

        fs_importance = []

        for w in self.meta_learner.coef_[0]:
            fs_importance = fs_importance + [w]*self.feature_split
    
        fs_importance = np.array(fs_importance)

        return biggest_fsse_fs, fs_importance

def fsse_homo_stacking(X, y, create_base_estimator_cv, split_range = range(1, 20), repeat = 3, summary = 'median', display = False):

    dic_accs = {}
    r = split_range
    N = repeat

    for split in r:

        for i in range(N):

            accs_repeat = []

            fsse = FSSE(create_base_estimator_cv, feature_split = split)
            fsse.fit(X, y)
            accs_repeat.append(fsse.evaluate(X, y))

        if summary == 'median':
            dic_accs[split] = np.median(accs_repeat)
        else:
            dic_accs[split] = np.mean(accs_repeat) # otherwise, mean

    if display:

        plt.figure(figsize = (12,3))
        plt.scatter(r, list(dic_accs.values()))
        plt.plot(r, list(dic_accs.values()))
        plt.show()
    
    return dic_accs

#endregion
