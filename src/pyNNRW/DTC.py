from sklearn.tree import DecisionTreeClassifier

def create_dtc_instance(L):
    return DecisionTreeClassifier(max_depth = L)