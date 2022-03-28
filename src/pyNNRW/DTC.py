from sklearn.tree import DecisionTreeClassifier

def create_dtc_instance(max_depth, criterion = 'gini', min_samples_split = 2, min_samples_leaf = 1):

    '''
    Create a decision tree classifier from sklearn.tree.DecisionTreeClassifier.

    Parameters
    ----------    
    criterion : {“gini”, “entropy”}, default=”gini”
    max_depth : max depth
    min_samples_split : default=2
    min_samples_leaf : default=1
    '''
    return DecisionTreeClassifier(max_depth = max_depth, criterion = criterion, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)


def create_stump_instance():
    '''
    Create a decision stump. The stump is a simpliest tree model with only one node (depth = 1).
    '''
    create_dtc_instance(max_depth = 1)