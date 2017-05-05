# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport cos, sin, sqrt

cpdef double[:,:] matrix_mult(double[:,:] A, double[:,:] B):
    return A.dot(B)

cpdef double[:,:] buildA(double[:,:] X, double[:,:] U1, double[:,:] V, double[:,:] U2):
    return np.dot(X, np.dot(U1, np.dot(V, U2)))


cpdef void _gradU(double[:] theta, int k,
                        int[:] ik, int[:] jk, int dim, double[:,:] V) nogil:
    """
    Gradient of a single Givens rotation
    """
    sin_tt = sin(theta[k])
    cos_tt = cos(theta[k])

    V[ik[k], ik[k]] = -sin_tt
    V[jk[k], ik[k]] = cos_tt
    V[ik[k], jk[k]] = -cos_tt
    V[jk[k], jk[k]] = -sin_tt


cpdef gradU(double[:] theta, int k,
            int[:] ik, int[:] jk, int dim):
    V = np.zeros((dim, dim))
    _gradU(theta, k, ik, jk, dim, V)
    return V


cpdef double[:,:] U_add_single(double[:,:] U1,
                  double[:] theta,
                  int k,
                  int[:] ik,
                  int[:] jk,
                  int dim) nogil:
    """
    add a single Givens rotation to a previous one (calc U1*U2)
    """
    cdef int col, ind_ik, ind_jk
    cdef double tt, u_ik
    tt = theta[k]
    cdef double cos_tt = cos(tt)
    cdef double sin_tt = sin(tt)
    cdef double[:,:] Uab = U1

    for col in range(dim):
        ind_ik = ik[k]
        ind_jk = jk[k]
        u_ik = Uab[ind_ik, col] * cos_tt - Uab[ind_jk, col] * sin_tt
        Uab[ind_jk, col] = Uab[ind_ik, col] * sin_tt + Uab[ind_jk, col] * cos_tt
        Uab[ind_ik, col] = u_ik

    return Uab

cpdef void build_Uab(double[:] theta, int a, int b,
               int[:] ik, int[:] jk, int dim, double[:,:] Uab):
    """
    Givens rotation for angles a to b
    """
    cdef int k, col,ind_ik,ind_jk
    cdef double tt, cos_tt, sin_tt, u_ik;

    if( b < a ):
        return

    for k in range(a, b+1):
        tt = theta[k]
        cos_tt = cos(tt)
        sin_tt = sin(tt)
        for col in range(dim):
            ind_ik = ik[k]
            ind_jk = jk[k]
            u_ik = Uab[ind_ik, col] * cos_tt - Uab[ind_jk, col] * sin_tt
            Uab[ind_jk, col] = Uab[ind_ik, col] * sin_tt + Uab[ind_jk, col] * cos_tt
            Uab[ind_ik, col] = u_ik


cpdef double[:,:] rotate_givens(double[:,:] X, double[:] theta,
                       int[:] ik, int[:] jk, int angle_num, int dim):
    """
    Rotate vectors in X with Givens rotation according to angles in theta
    """
    G = np.eye(dim)
    build_Uab(theta, 0, angle_num-1, ik, jk, dim, G);
    return np.dot(X, G)


cpdef double evqualitygrad(double[:,:] X, double[:] theta,
                     int[:] ik, int[:] jk,
                     int angle_num, int angle_index,
                     int dim, int ndata,
                     double[:,:] V, double[:,:] U1, double[:,:] U2,
                     double[:] max_values, int[:] max_index):

    cdef double dJ=0, tmp1, tmp2, mvsq, mvcb;

    # build V,U,A
    _gradU(theta,angle_index,ik,jk,dim,V)
    # print("Computed gradU")
    # print(np.asarray(V))

    # U1 = ...
    build_Uab(theta,0,angle_index-1,ik,jk,dim,U1)

    # U2 = ...
    build_Uab(theta,angle_index+1,angle_num-1,ik,jk,dim,U2)
    # print("Computed Uab")
    # print(np.asarray(U1))
    # print(np.asarray(U2))

    cdef double[:,:] A = buildA(X, U1, V, U2)
    # print("builtA")
    # print(np.asarray(A))

    # rotate vecs according to current angles */
    cdef double[:,:] Y = rotate_givens(X,theta,ik,jk,angle_num,dim)
    # print ("Rotated according to Givens successfully")

    # find max of each row
    findrowmax(Y, max_values, max_index, 0)
    # print("Found max of each row")
    # print(np.asarray(max_values)[:10])

    # compute gradient
    for i in range(ndata): # loop over all rows
        mvsq = max_values[i]*max_values[i]
        mvcb = mvsq * max_values[i]
        for j in range(dim):  # loop over all columns
            tmp1 = A[i,j] * Y[i,j] / (mvsq)
            tmp2 = A[i,max_index[i]] * Y[i,j]* Y[i,j] / mvcb
            dJ += tmp1-tmp2;
            # print("i = {}, j = {}, tmp1 = {}, tmp2 = {}".format(i,j,tmp1,tmp2))

    dJ = 2*dJ/ndata/dim;
    # print("Computed gradient = {}".format(dJ))
    return dJ


cpdef double evqual(double[:,:] X,
              int[:] ik, int[:] jk,
              int dim, int ndata,
              double[:] max_values, int[:] max_index):
    """
    alignment quality
    """

    findrowmax(X, max_values, max_index, 1)
    # print("found max of each row")

    # compute cost
    cdef double J=0

    for i in range(ndata):
        for j in range(dim):
            J += X[i,j]*X[i,j]/max_values[i]

    J = 1.0 - (J/ndata -1.0)/dim
    # print("Computed quality = {}".format(J))
    return J


cpdef cluster_assign(double[:,:] X,
                        int[:] ik, int[:] jk,
                        int dim, int ndata):
    """
    cluster assignments
    """
    max_index = np.argmax(np.abs(X), axis=1)
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


cpdef evrot(double[:,:] X, int method):
    # print("just starting")

    # get the number and length of eigenvectors dimensions */
    cdef int ndata = X.shape[0]
    cdef int dim = X.shape[1]

    # get the number of angles
    cdef int angle_num
    angle_num = (dim * (dim-1)) /2
    cdef double[:] theta = np.zeros(angle_num)
    # print("Angle number is {}".format(angle_num))

    # build index mapping
    cdef int i,j,k=0;
    tri_ix = np.triu_indices(dim, 1)
    cdef int[:] ik = np.zeros(angle_num, dtype=np.intc)
    cdef int[:] jk = np.zeros(angle_num, dtype=np.intc)
    for i in range(0, dim):
        for j in range(i+1, dim):
            ik[k] = i
            jk[k] = j
            k += 1

    # print("Built index mapping for {} angles".format(k))

    # definitions
    cdef int max_iter = 200
    cdef double dQ, Q, Q_new, Q_old1, Q_old2, Q_up, Q_down
    cdef double alpha
    cdef int iter, d
    cdef double[:,:] Xrot = X
    cdef double[:] theta_new = theta

    cdef double[:,:] V = np.zeros((dim,dim), dtype=np.double)
    cdef double[:,:] U1 = np.zeros((dim,dim), dtype=np.double)
    cdef double[:,:] U2 = np.zeros((dim,dim), dtype=np.double)
    cdef double[:] max_values = np.zeros(ndata, dtype=np.double)
    cdef int[:] max_index = np.zeros(ndata, dtype=np.intc)

    Q = evqual(X, ik, jk, dim, ndata, max_values, max_index); # initial quality
    # print("Q = {}".format(Q))
    Q_old1 = Q
    Q_old2 = Q
    iter = 0

    while( iter < max_iter ): # iterate to refine quality
        iter += 1
        for d in range(angle_num):
            if method == 2: # descend through numerical derivative
                alpha = 0.1
                # move up */
                theta_new[d] = theta[d] + alpha
                Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim)
                Q_up = evqual(Xrot,ik,jk,dim,ndata,max_values, max_index)

                # move down */
                theta_new[d] = theta[d] - alpha
                Xrot = rotate_givens(X, theta_new, ik, jk, angle_num, dim)
                Q_down = evqual(Xrot, ik, jk, dim, ndata,
                                max_values, max_index)

                # update only if at least one of them is better */
                if (Q_up > Q) or (Q_down > Q):
                    if Q_up > Q_down:
                        theta[d] = theta[d] + alpha
                        theta_new[d] = theta[d]
                        Q = Q_up
                    else:
                        theta[d] = theta[d] - alpha
                        theta_new[d] = theta[d]
                        Q = Q_down

            else: # descend through true derivative
                alpha = 1.0;
                V[:] = 0
                U1[:] = 0
                U2[:] = 0
                for i in range(dim):
                    U1[i,i] = U2[i,i] = 1
                dQ = evqualitygrad(X,theta,ik,jk,angle_num,d,dim,ndata,V,U1,U2,max_values,max_index)
                theta_new[d] = theta[d] - alpha * dQ
                # print("theta new = {}".format(theta_new[d]))
                Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim)
                Q_new = evqual(Xrot,ik,jk,dim,ndata,max_values, max_index)
                if Q_new > Q:
                    theta[d] = theta_new[d]
                    Q = Q_new

                else:
                    theta_new[d] = theta[d]

        # stopping criteria
        if iter > 2:
            if (Q - Q_old2) < 1e-3:
                break

        Q_old2 = Q_old1
        Q_old1 = Q

    # print("Done after {} iterations, Quality is {}".format(iter,Q))
    Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim)
    clusts = cluster_assign(Xrot,ik,jk,dim,ndata)

    # prepare output **/
    return (clusts, Q, np.asarray(Xrot))


cpdef int findrowmax(double[:,:] arr, double[:] maxes, int[:] index, int store_square) nogil:
    """
    Finds max value in each row of `arr`, storing the value in `maxes` and
    its column index in `index`
    """
    cdef int nrow = arr.shape[0]
    cdef int ncol = arr.shape[1]
    if not nrow == maxes.shape[0] or not nrow == index.shape[0]:
        return -1

    cdef int i
    cdef double what = -INFINITY
    cdef int where = -1
    cdef double val = 0
    for i in range(nrow):
        for j in range(ncol):
            val = arr[i,j] * arr[i,j]
            if val > what:
                what = val
                where = j
        if store_square:
            maxes[i] = what
        else:
            maxes[i] = arr[i, where]
        index[i] = where
        what = -INFINITY
        where = -1

    return 0