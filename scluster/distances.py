import numpy as np

def kdists(matrix, k=7, ix=None):
    """ Returns the k-th nearest distances, row-wise, as a column vector """

    ix = ix or kindex(matrix, k)
    return matrix[ix][np.newaxis].T


def kindex(matrix, k):
    """ Returns indices to select the kth nearest neighbour"""

    ix = (np.arange(len(matrix)), matrix.argsort(axis=0)[k])
    return ix


def kscale(matrix, k=7, dists=None):
    """ Returns the local scale based on the k-th nearest neighbour """
    dists = (kdists(matrix, k=k) if dists is None else dists)
    scale = dists.dot(dists.T)
    return scale

def affinity(matrix, scale):
    msq = matrix * matrix
    scaled = -msq / scale
    scaled[np.where(np.isnan(scaled))] = 0.0
    a = np.exp(scaled)
    a.flat[::matrix.shape[0]+1] = 1.0
    return a

def laplace(affinity_matrix, shi_malik_type=False):
    """ Converts affinity matrix into normalised graph Laplacian,
    for spectral clustering.
    (At least) two forms exist:

    L = (D^-0.5).A.(D^-0.5) - default

    L = (D^-1).A - `Shi-Malik` type, from Shi Malik paper"""

    diagonal = affinity_matrix.sum(axis=1) - affinity_matrix.diagonal()
    zeros = diagonal <= 1e-10
    diagonal[zeros] = 1
    if (diagonal <= 1e-10).any():  # arbitrarily small value
        raise ZeroDivisionError
    if shi_malik_type:
        inv_d = np.diag(1 / diagonal)
        return inv_d.dot(affinity_matrix)
    diagonal = np.sqrt(diagonal)
    return affinity_matrix / diagonal / diagonal[:, np.newaxis]

def eigen(matrix):
    """ Calculates the eigenvalues and eigenvectors of the input matrix.
    Returns a tuple of (eigenvalues, eigenvectors, cumulative percentage of
    variance explained). Eigenvalues and eigenvectors are sorted in order of
    eigenvalue magnitude, high to low """

    (vals, vecs) = np.linalg.eigh(matrix)
    ind = vals.argsort()[::-1]
    vals = vals[ind]
    vecs = vecs[:, ind]
    vals_ = vals.copy()
    vals_[vals_ < 0] = 0.
    return vals, vecs

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