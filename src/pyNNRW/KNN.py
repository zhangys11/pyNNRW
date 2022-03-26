from sklearn.neighbors import KNeighborsClassifier

def create_knn_instance(L):
    return KNeighborsClassifier(n_neighbors = L)