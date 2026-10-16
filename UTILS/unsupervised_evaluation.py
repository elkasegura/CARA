import numpy as np
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        raise ValueError("L1.shape must == L2.shape")

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class), dtype=np.int64)

    for i in range(n_class1):
        for j in range(n_class2):
            ss = (l1 == label1[i])
            tt = (l2 == label2[j])
            G[i, j] = np.count_nonzero(ss & tt)

    # Hungarian algorithm (maximize matches -> minimize -G)
    row_ind, col_ind = linear_sum_assignment(-G)

    new_l2 = np.zeros_like(l2)
    # keep the same mapping logic: map predicted label2 -> true label1
    for i in range(n_class2):
        new_l2[l2 == label2[col_ind[i]]] = label1[row_ind[i]]

    return new_l2.astype(int)


def evaluation(X_selected, n_clusters, y):
    """
    This function calculates ACC and NMI of clustering results
    """
    k_means = KMeans(
        n_clusters=n_clusters,
        init="k-means++", #"k-means++"
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
    )

    k_means.fit(X_selected)
    y_predict = k_means.labels_

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return nmi, acc
