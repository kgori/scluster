# scluster
Implementation of self-tuning spectral clustering from Zelnik-Manor and Perona (2004)

Example usage: sclust.py [--scale INT] distances.csv min=2 max=12

In the original paper the local scaling parameter was set to 7 - i.e. the distance to the 7th nearest neighbour. The --scale parameter selects the K-th nearest neighbour.
