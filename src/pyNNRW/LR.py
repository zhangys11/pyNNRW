from sklearn.linear_model import LogisticRegressionCV

def create_lr_instance(L):
    return LogisticRegression(C = L)