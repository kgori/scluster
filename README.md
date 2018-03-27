# scluster
An implementation in Python of the self-tuning spectral clustering algorithm from Zelnik-Manor and Perona (2004).
The original Matlab implementation can be found [here](https://lihi.eew.technion.ac.il/files/Demos/SelfTuningClustering.html "Self-tuning spectral clustering project page").

Example usage: sclust.py [--scale INT] distances.csv min=2 max=12

In the original paper the local scaling parameter was set to 7 - i.e. the distance to the 7th nearest neighbour. The `scale` parameter selects the K-th nearest neighbour.

`min` and `max` select the minimum and maximum number of clusters to return, respectively. Note that the algorithm slows down noticeably once it hits around 50 clusters.
