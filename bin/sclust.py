#!/usr/bin/env python
from __future__ import print_function

import os
import sys

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scluster import distances
from scluster import cluster


def filecheck(filename):
    if not os.path.exists(filename):
        print("Quitting: Could not find file {}".format(filename))
        sys.exit()


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
    print(result)
