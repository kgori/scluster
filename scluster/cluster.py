from __future__ import print_function

"""
Implements the self-tuning by roation of eigenvectors algorithm of Zelnik-Manor, Lihi, and Pietro Perona. 2004.
"Self-Tuning Spectral Clustering." In Advances in Neural Information Processing Systems, 1601-8.
"""
import numpy as np
from scluster.distances import eigen, laplace
from evrot import evrot

class Result(object):

    def __init__(self, assignments, memberships, number, scores):
        self.assignments = assignments
        self.memberships = memberships
        self.number = number
        self.vectors = scores

    def __str__(self):
        return ("Found {} clusters:\n"
                "Cluster assignments:\n\t{}\n"
                "Cluster memberships:\n\t{}".format(self.number,
                                                    self.assignments,
                                                    [a.tolist() for a in self.memberships]))

def order(l):
    """
    This function preserves the group membership,
    but sorts the labelling into numerical order
    """
    from collections import defaultdict

    list_length = len(l)

    d = defaultdict(list)
    for (i, element) in enumerate(l):
        d[element].append(i)

    l2 = [None] * list_length

    for (name, index_list) in enumerate(sorted(d.values(), key=min),
                                        start=1):
        for index in index_list:
            l2[index] = name

    return tuple(l2)


def cluster_rotate(A, group_num=None, method=None):
    """
    cluster by rotating eigenvectors to align with the canonical coordinate
    system

    :param A: Affinity matrix
    :param group_num: an array of group numbers to test
                      it is assumed to be a continuous set
    :param method:    method - 1   gradient descent
                   2   approximate gradient descent
    :return: clusts - a cell array of the results for each group number
             best_group_index - the group number index with best alignment quality
             Quality = the final quality for all tested group numbers
             Vr = the rotated eigenvectors
    """
    if group_num is None:
        group_num = [2,3,4,5,6]

    if method is None:
        method = 1  # change to any other value to estimate gradient numerically

    group_num = [x for x in sorted(group_num) if not x == 1]

    nclusts = max(group_num)
    evals, V = eigen(laplace(A), nclusts)

    # Rotate eigenvectors
    Vr = [None] * len(group_num)  # List holding best rotated eigenvectors for each iteration
    Vbuffer = np.zeros_like(V)  # Buffer holding rotated eigenvectors (changes each iteration)
    Vbuffer[:, :group_num[0] - 1] = V[:, :group_num[0] - 1]
    # rotated = [None] * len(group_num)
    clusts = [None] * len(group_num)
    quality = np.zeros(len(group_num))

    # for g in range(len(group_num)):
    for i, g in enumerate(group_num):
        print ("Testing {} clusters...".format(group_num[i]))
        Vcurr = Vbuffer[:, :g]
        Vcurr[:, g - 1] = V[:, g - 1]
        clusts[i], quality[i], Vbuffer[:, :g] = evrot(Vcurr, method)
        Vr[i] = Vbuffer[:, :g].copy()

    i = np.where(quality.max() - quality < 0.001)[0]
    best_group_index = i[-1]

    return clusts, best_group_index, quality, Vr