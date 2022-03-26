from .ELM import *
from .RVFL import *
from .MLP import *
from .DTC import *
from .LR import *
from .KNN import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from keras.utils import to_categorical
import numpy as np
import time


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
        
    train_loss, train_acc = model.evaluate(x_train, t_train)
    if (verbose):
        print('train_loss: %f' % train_loss) # loss value
        print('train_acc: %f' % train_acc) # accuracy
        print('train_precision: %f' % train_precision)
        print('train_recall: %f' % train_recall) # uar (unweighted average recall)


    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc = model.evaluate(x_test, t_test)
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

