from __future__ import print_function
"""
Implements the self-tuning by roation of eigenvectors algorithm of Zelnik-Manor, Lihi, and Pietro Perona. 2004.
“Self-Tuning Spectral Clustering.” In Advances in Neural Information Processing Systems, 1601–8.
"""
import numba
import numpy as np
from math import cos, sin

def sum_dJ(A_x_Y, Y_sq, mv_sq, mv_cb, max_A_values, dim, ndata):
    return 2 * np.mean((A_x_Y / mv_sq[:,np.newaxis]) -
                       (max_A_values[:,np.newaxis] * (Y_sq / mv_cb[:,np.newaxis])))

@numba.jit
def build_Uab(theta, a, b, ik, jk, dim):
    """
    Not as fast as pure C, but close enough
    """
    Uab = np.eye(dim, dtype=np.double)
    if b < a:
        return Uab

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for k in range(a, b+1):
        m1 = ik[k]
        m2 = jk[k]

        for n in range(dim):
            tmp = Uab[n,m1] * cos_theta[k] - Uab[n,m2] * sin_theta[k]
            Uab[n,m2] = Uab[n,m1] * sin_theta[k] + Uab[n,m2] * cos_theta[k]
            Uab[n,m1] = tmp

    return Uab

def buildA(X, U1, Vk, U2):
    # A = X*U1*Vk*U2
    A = np.dot(X, np.dot(U1, np.dot(Vk, U2)))
    return A

def gradU(theta, k, ik, jk, dim):
    V = np.zeros((dim, dim), dtype=np.float)
    i = ik[k]
    j = jk[k]
    tt = theta[k]
    sin_tt = sin(tt)
    cos_tt = cos(tt)

    V[[i, i, j, j], [i, j, i, j]] = [-sin_tt, cos_tt, -cos_tt, -sin_tt]

    return V

def rotate_givens(X, theta, ik, jk, angle_num, dim):
    """
    Rotates matrix
    """
    G = build_Uab(theta, 0, angle_num - 1, ik, jk, dim)
    Y = np.dot(X, G)
    return Y

def evqualitygrad(X, theta, ik, jk, angle_num, angle_index, dim, ndata):
    """
    Gradient of eigenvector quality
    """
    V = gradU(theta, angle_index, ik, jk, dim)

    U1 = build_Uab(theta, 0, angle_index - 1, ik, jk, dim)
    U2 = build_Uab(theta, angle_index + 1, angle_num - 1, ik, jk, dim)

    A = buildA(X, U1, V, U2)

    Y = rotate_givens(X, theta, ik, jk, angle_num, dim)

    # Find max of each row
    r_ndata = range(ndata)

    Y_sq = Y ** 2
    max_index = np.argmax(Y_sq, axis=1)
    max_values = Y[r_ndata, max_index]

    max_A_values = A[r_ndata, max_index]
    mv_sq = max_values * max_values
    mv_cb = max_values * mv_sq
    A_x_Y = A * Y

    # Compute gradient
    dJ = sum_dJ(A_x_Y, Y_sq, mv_sq, mv_cb, max_A_values, dim, ndata)

    return dJ

def evqual(X):
    """
    Eigenvector quality (diagonal-ness)
    """
    Xsquare = X * X
    max_values = Xsquare.max(axis=1)[:, np.newaxis]

    # Compute cost
    Xsq_div_maxvals = Xsquare / max_values
    J = 1 - np.mean(Xsq_div_maxvals) + 1.0 / X.shape[1]

    return J

def cluster_assign(X):  # , ik, jk, dim, ndata):
    """
    Assign data points to clusters
    """
    (ndata, dim) = X.shape
    Xsq = X * X
    max_index = np.argmax(Xsq, axis=1)
    cluster_count = np.bincount(max_index, minlength=dim)
    cluster_cell_array = [np.array([0] * count) for count in cluster_count]

    for j in range(dim):
        cluster = cluster_cell_array[j]
        cind = 0
        for i in range(ndata):
            if max_index[i] == j:
                cluster[cind] = i + 1
                cind += 1
    #return [cell for cell in cluster_cell_array if len(cell) > 0]
    # downstream code handles collapsing empty clusters
    return cluster_cell_array

def test(X):
    (ndata, dim) = X.shape
    (ik, jk) = np.triu_indices(dim, k=1)
    angle_num = len(ik)

    theta = np.arange(0.0, angle_num / 10., 0.1)
    Q = evqual(X)
    #print('Q = {0}'.format(Q))
    dQ = evqualitygrad(X, theta, ik, jk, 45, 5, 10, 40)
    #print('x:', X)
    #print('theta:', theta)
    #print('ik and jk', ik, jk)
    #print('Q:', Q)
    #print('dQ:', dQ)

def evrot(X, max_iter=200):
    """
    Workhorse of eigenvector rotation
    """
    ndata, dim = X.shape
    ik, jk = np.triu_indices(dim, k=1)
    angle_num = ik.size

    theta = np.zeros(angle_num, dtype=np.double)
    theta_new = np.zeros_like(theta)

    Q = evqual(X)
    Q_old1 = Q
    Q_old2 = Q
    alpha = 1
    for iteration in range(max_iter):
        for d in range(angle_num):
            dQ = evqualitygrad(X, theta, ik, jk, angle_num, d, dim, ndata)
            theta_new[d] = theta[d] - alpha * dQ
            Xrot = rotate_givens(X, theta_new, ik, jk, angle_num, dim)
            Q_new = evqual(Xrot)
            if Q_new > Q:
                theta[d] = theta_new[d]
                Q = Q_new
            else:
                theta_new[d] = theta[d]

        # Stopping criterion
        if iteration > 1:
            if Q - Q_old2 < 0.001:
                break

        Q_old2 = Q_old1
        Q_old1 = Q

    Xrot = rotate_givens(X, theta_new, ik, jk, angle_num, dim)
    clusts = cluster_assign(Xrot)

    return clusts, Q, Xrot

def cluster_rotate(coordinates, min_groups=2, max_groups=None, thresh=0.01):
    """
    Clusters data points by rotating eigenvectors (i.e. coordinates)

    :param coordinates: numpy array of coordinates, dimensions [datapoints x dimensions]
    :param min_groups: minimum number of clusters to consider (imposes lower limit of 2)
    :param max_groups: maximum number of clusters to consider (imposes upper limit of n datapoints)
    :return: tuple: (number of groups, group memberships, quality scores, rotated coordinates)
    """
    if max_groups is None:
        max_groups = coordinates.shape[0]

    max_groups = min(coordinates.shape[0], max_groups)
    min_groups = max(2, min_groups)

    groups = list(range(min_groups, max_groups + 1))
    vector_length = coordinates.shape[0]
    current_vector = coordinates[:, :groups[0]]
    n = max_groups - min_groups + 1

    quality_scores = [None] * n
    clusters = [None] * n
    rotated_vectors = [None] * n

    print("Beginning cluster selection")
    for g in range(n):
        print("Ngroups={}".format(min_groups + g))
        if g > 0:
            current_vector = np.concatenate((rotated_vectors[g - 1],
                                             coordinates[:, groups[g] - 1:groups[g]]), axis=1)

        (clusters[g], quality_scores[g], rotated_vectors[g]) = \
            evrot(current_vector)

    # Find the highest index of quality scores where the
    # score is within `thresh` of the maximum:
    # this is our chosen number of groups

    max_score = max(quality_scores)
    index = quality_scores.index(max_score)
    start = index + 1
    for (i, score) in enumerate(quality_scores[index + 1:], start=start):
        if abs(score - max_score) < thresh:
            index = i

    return (groups[index], clusters[index], quality_scores,
            rotated_vectors[index])
