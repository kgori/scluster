#!/usr/bin/env python
from __future__ import print_function

import os
import sys

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scluster import distances
from scluster import cluster


def filecheck(filename):
    if not os.path.exists(filename):
        print("Quitting: Could not find file {}".format(filename))
        sys.exit()

def assn_to_p(assignments):
    """
    Convert list of assignments to partition list

    list of assignments (example):
    [ array(10,15,6), array(1,4,5,11,14), array(2,3,7,8,9,12,13)]

    converts to
    [1,2,2,1,1,0,2,2,2,0,1,2,2,1,0]

    equivalent to (minimum 1-based representation)
    [1,2,2,1,1,3,2,2,2,3,1,2,2,1,3]

    """
    n_elem = max(np.max(arr) for arr in assignments)
    min_elem = min(np.min(arr) for arr in assignments)
    plist = np.zeros(n_elem)
    for i, arr in enumerate(assignments):
        plist[arr-min_elem] = i

    return cluster.order(plist)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--distances', help="File of distance matrix (assumes CSV)")
    parser.add_argument('-c', '--coordinates', help="File of coordinates (assumes CSV)")
    parser.add_argument('--auto', nargs=2, type=int, default=[2, 6], metavar=('MIN', 'MAX'),
                        help="Automatically select the number of clusters between --args=MIN MAX values")
    parser.add_argument('-n', '--number', type=int, help="Fix this number of clusters")
    parser.add_argument('-s', '--scale', type=int, default=7, help="Local scaling parameter (default 7)")
    parser.add_argument('-m', '--method', type=int, choices=[1,2], default=1,
                        help="Optimisation method: 1 - analytical gradient descent (default), 2 - approximate gradient descent")
    parser.add_argument('-p', '--plot', action='store_true', help='Use matplotlib to generate a plot (requires coordinate input)')
    args = parser.parse_args()

    if args.distances:
        filecheck(args.distances)
        dists = pd.DataFrame.from_csv(args.distances).values
    elif args.coordinates:
        filecheck(args.coordinates)
        coords = pd.DataFrame.from_csv(args.coordinates).values
        dists = squareform(pdist(coords))
    else:
        print("Need either distances or coordinates as input")
        sys.exit()

    aff = distances.affinity(dists, distances.kscale(dists, args.scale))

    if args.number:
        min_ = args.number
        max_ = args.number

    elif args.auto:
        if not args.auto[1] > args.auto[0]:
            print("Maximum number of clusters is less than minimum. Try again.")
            sys.exit()

        min_ = args.auto[0]
        max_ = args.auto[1]

    groups = list(range(min_, max_ + 1))
    result = cluster.cluster_rotate(aff, groups, args.method)
    # print(result)

    best_index = result[1]
    chosen_groups = groups[best_index]
    quality_scores = result[2][best_index]
    best_rotation = result[3][best_index]
    best_assignment = result[0][best_index]

    print("Determined number of clusters to be {}".format(chosen_groups))
    print("Quality scores = {}".format(quality_scores))
    print("Assignment = {}".format(assn_to_p(best_assignment)))

    if args.plot and args.coordinates:
        import matplotlib.pyplot as plt
        plt.scatter(*coords[:, :2].T, c=np.array(assn_to_p(best_assignment)))
        plt.show()
