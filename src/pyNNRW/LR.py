from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def create_lr_instance(C = 1.0, penalty = 'l2', multi_class = 'auto'):

    '''
    Create an sklearn.linear_model.LogisticRegression instance.

    Parameters
    ----------
    penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
    C : Inverse of regularization strength
    multi_class : {‘auto’, ‘ovr’, ‘multinomial’}
    '''
    return LogisticRegression(C = C, penalty = penalty, multi_class = multi_class)