# pyNNRW

pyNNRW: A Python library for NNRW (neural network with random weights).   

Basic functions:
1. Implements 2 fundamental NNRW flavors, i.e., ELM and RVFL.  
2. Performance comparison with main machine learning models, e.g., SVM, decision tree, MLP.   
3. NNRW-based ensembles (in progress).

# Publication

Spectroscopic Profiling-based Geographic Herb Identification by Neural Network with Random Weights [J]. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2022, doi: 10.1016/j.saa.2022.121348

# Installation

> pip install pyNNRW

# How to use

Download the sample dataset from the /data folder.
There are two data files: 1. 7044X_RAMAN.csv 2. 7143X_UV.csv.  
The two files are the Raman and UV (ultra-violet) spectroscopic profiling data of herb samples from 4 different regions.  

Use the following sample code to use the package:

<1> Use low-level classes, e.g., ELM, RVFL. The following code trains and tests an ELM model.

    # ===============================
    # Import library
    # ===============================
    # import the library
    from pyNNRW import elm, rvfl

    # ===============================
    # Load dataset
    # ===============================
    df = pd.read_csv('7044X_RAMAN.csv')
    X = np.array(df.iloc[:,1:])
    y = np.array(df.iloc[:,0]) # 1st col is the label
    n_classes = len(set(y))
    x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.2)
    t_train = to_categorical(t_train, n_classes).astype(np.float32)
    t_test = to_categorical(t_test, n_classes).astype(np.float32)
    # print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

    # ===============================
    # set ELM parameters
    # ===============================
    n_hidden_nodes = L #x_train.shape[1]
    loss = 'mean_squared_error' # 'mean_absolute_error'
    activation = 'sigmoid' # 'identity'

    # ===============================
    # Instantiate ELM
    # ===============================
    model = elm.ELM( # or rvfl.RVFL
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
        
    train_loss, train_acc, train_precision, train_recall = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy', 'precision', 'recall'])

    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc, val_precision, val_recall = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy', 'precision', 'recall'])


<2> You may also use high-level APIs, as follows.

    from pyNNRW import nnrw

    # train and test an ELM model
    train_acc, val_acc, t = nnrw.ELMClf(X, y, L = 20, verbose = False) # L is hidden layer nodes

    # train and test a RVFL model
    train_acc, val_acc, t = nnrw.RVFLClf(X, y, L = 20, verbose = False) # L is hidden layer nodes

    # Conduct a performance test for ELM at varied L hyper-parameters (1~60). Each iteration is averaged on 20 rounds.
    train_accs, val_accs, ts = nnrw.PerformenceTests(ELMClf, X, y, Ls = list(range(1, 60)), N = 20)

# New function in v0.2.0

We added Kernel-NNRW, which provides a series of kernels combined with NNRW.
