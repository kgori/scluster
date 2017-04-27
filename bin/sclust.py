#!/usr/bin/env python
from __future__ import print_function

import os
import sys

import pandas as pd
from scluster import distances
from scluster import cluster

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('distances', help="CSV file of distance matrix")
    parser.add_argument('min', type=int, help="Minimum number of clusters to search for")
    parser.add_argument('max', type=int, help="Maximum number of clusters to search for")
    parser.add_argument('-s', '--scale', type=int, default=7, help="Local scaling parameter (default 7)")
    args = parser.parse_args()

    if not os.path.exists(args.distances):
        print("Quitting: Could not find input file {}".format(args.distances))
        sys.exit()

    dists = pd.DataFrame.from_csv(args.distances)
    coords = distances.distances_to_coordinates(dists.values, dists.shape[1])
    result = cluster.spectral_rotate(coords, min_groups=args.min, max_groups=args.max)
    print(result)
