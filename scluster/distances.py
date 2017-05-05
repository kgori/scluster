import numpy as np

def kdists(matrix, k=7, ix=None):
    """ Returns the k-th nearest distances, row-wise, as a column vector """

    ix = ix or kindex(matrix, k)
    return matrix[ix][np.newaxis].T


def kindex(matrix, k):
    """ Returns indices to select the kth nearest neighbour"""

    ix = (np.arange(len(matrix)), matrix.argsort(axis=0)[k])
    return ix


def kscale(matrix, k=7, dists=None, minval=0.004):
    """ Returns the local scale based on the k-th nearest neighbour """
    dists = (kdists(matrix, k=k) if dists is None else dists)
    scale = dists.dot(dists.T)
    return np.clip(scale, minval, np.inf)


def affinity(matrix, scale):
    msq = matrix * matrix
    scaled = -msq / scale
    scaled[np.where(np.isnan(scaled))] = 0.0
    a = np.exp(scaled)
    a.flat[::matrix.shape[0]+1] = 0.0  # zero out the diagonal
    return a


def laplace(affinity_matrix):
    """
    Converts affinity matrix into normalised graph Laplacian,
    for spectral clustering.

    L = (D^-0.5).A.(D^-0.5)
    """

    diagonal = affinity_matrix.sum(axis=1) + np.finfo(np.double).eps
    diagonal = 1.0 / np.sqrt(diagonal)
    return diagonal[:, np.newaxis] * affinity_matrix * diagonal


def normalise_rows(matrix):
    """ Scales all rows to length 1. Fails when row is 0-length, so it
    leaves these unchanged """

    lengths = np.apply_along_axis(np.linalg.norm, 1, matrix)
    if not (lengths > 0).all():
        # raise ValueError('Cannot normalise 0 length vector to length 1')
        # print(matrix)
        lengths[lengths == 0] = 1
    return matrix / lengths[:, np.newaxis]


def laplace_to_coordinates(laplacian, dimensions):
    _, vecs = eigen(laplacian)
    return vecs[:, :dimensions]


def distances_to_coordinates(dm, dim):
    scale = kscale(dm, 7)
    aff = affinity(dm, scale)
    lap = laplace(aff)
    return laplace_to_coordinates(lap, dim)


def eigen(matrix, n):
    """ Calculates the eigenvalues and eigenvectors of the input matrix.
    Returns a tuple of (eigenvalues, eigenvectors, cumulative percentage of
    variance explained). Eigenvalues and eigenvectors are sorted in order of
    eigenvalue magnitude, high to low """

    (vals, vecs) = np.linalg.eigh(matrix)
    ind = vals.argsort()[::-1]
    vals = vals[ind]
    vecs = vecs[:, ind]
    return vals[:n], vecs[:,:n]
