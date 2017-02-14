
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass


def munkres(costMat):

    # Local Variables: validCol, validRow, coverRow, colIdx, pass, cost, bigM, costMat, uZc, rowZ1, starZ, nCols, minval, cC, minC, maxv, nRows, Step, primeZ, validMat, cR, minR, zP, A, assignment, vIdx, rIdx, coverColumn, a, c, b, rowIdx, n, cIdx, r, stz, dMat, z, uZr
    # Function calls: disp, rand, all, false, munkres, find, size, outerplus, min, diag, sum, bsxfun, any, zeros, toc, log10, tic, trace, max, ceil, ones, isempty, true, magic, Inf
    #% MUNKRES   Munkres (Hungarian) Algorithm for Linear Assignment Problem.
    #%
    #% [ASSIGN,COST] = munkres(COSTMAT) returns the optimal column indices,
    #% ASSIGN assigned to each row and the minimum COST based on the assignment
    #% problem represented by the COSTMAT, where the (i,j)th element represents the cost to assign the jth
    #% job to the ith worker.
    #%
    #% Partial assignment: This code can identify a partial assignment is a full
    #% assignment is not feasible. For a partial assignment, there are some
    #% zero elements in the returning assignment vector, which indicate
    #% un-assigned tasks. The cost returned only contains the cost of partially
    #% assigned tasks.
    #% This is vectorized implementation of the algorithm. It is the fastest
    #% among all Matlab implementations of the algorithm.
    #% Examples
    #% Example 1: a 5 x 5 example
    #%{
    [assignment, cost] = munkres(magic(5.))
    np.disp(assignment)
    #% 3 2 1 5 4
    np.disp(cost)
    #%15
    #%}
    #% Example 2: 400 x 400 random data
    #%{
    n = 400.
    A = np.random.rand(n)
    tic
    [a, b] = munkres(A)
    toc
    #% about 2 seconds
    #%}
    #% Example 3: rectangular assignment with inf costs
    #%{
    A = np.random.rand(10., 7.)
    A[int((A > 0.7)) - 1] = Inf
    [a, b] = munkres(A)
    #%}
    #% Example 4: an example of partial assignment
    #%{
    A = np.array(np.vstack((np.hstack((1., 3., Inf)), np.hstack(
        (Inf, Inf, 5.)), np.hstack((Inf, Inf, 0.5)))))
    [a, b] = munkres(A)
    #%}
    #% a = [1 0 3]
    #% b = 1.5
    #% Reference:
    #% "Munkres' Assignment Algorithm, Modified for Rectangular Matrices",
    #% http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
    #% version 2.3 by Yi Cao at Cranfield University on 11th September 2011
    assignment = np.zeros(1., matcompat.size(costMat, 1.))
    cost = 0.
    validMat = np.logical_and(costMat == costMat, costMat < Inf)
    bigM = 10.**(np.ceil(np.log10(np.sum(costMat[int(validMat) - 1]))) + 1.)
    costMat[int((not validMat)) - 1] = bigM
    #% costMat(costMat~=costMat)=Inf;
    #% validMat = costMat<Inf;
    validCol = np.any(validMat, 1.)
    validRow = np.any(validMat, 2.)
    nRows = np.sum(validRow)
    nCols = np.sum(validCol)
    n = matcompat.max(nRows, nCols)
    if not n:
        return []

    maxv = 10. * matcompat.max(costMat[int(validMat) - 1])
    dMat = np.zeros(n) + maxv
    dMat[0:nRows, 0:nCols] = costMat[int(validRow) - 1, int(validCol) - 1]
    #%*************************************************
    #% Munkres' Assignment Algorithm starts here
    #%*************************************************
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%   STEP 1: Subtract the row minimum from each row.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minR = matcompat.max(dMat, np.array([]), 2.)
    minC = matcompat.max(bsxfun(minus, dMat, minR))
    #%**************************************************************************
    #%   STEP 2: Find a zero of dMat. If there are no starred zeros in its
    #%           column or row start the zero. Repeat for each zero
    #%**************************************************************************
    zP = dMat == bsxfun(plus, minC, minR)
    starZ = np.zeros(n, 1.)
    while np.any(zP.flatten(1)):
        [r, c] = nonzero(zP, 1.)
        starZ[int(r) - 1] = c
        zP[int(r) - 1, :] = false
        zP[:, int(c) - 1] = false

    while 1.:
        #%**************************************************************************

        #% Cost of assignment
    rowIdx = nonzero(validRow)
    colIdx = nonzero(validCol)
    starZ = starZ[0:nRows]
    vIdx = starZ <= nCols
    assignment[int(rowIdx[int(vIdx) - 1]) - 1] = colIdx[int(starZ[int(vIdx) - 1]) - 1]
    pass = assignment[int((assignment > 0.)) - 1]
    pass[int((not np.diag[int(validMat[int((assignment > 0.)) - 1, int(pass) - 1]) - 1])) - 1] = 0.
    assignment[int((assignment > 0.)) - 1] = pass
    cost = np.trace(costMat[int((assignment > 0.)) - 1,
                            int(assignment[int((assignment > 0.)) - 1]) - 1])
    return [assignment, cost]


def outerplus(M, x, y):

    # Local Variables: c, minval, M, cIdx, ny, rIdx, y, x
    # Function calls: inf, min, outerplus, find, size
    ny = matcompat.size(M, 2.)
    minval = np.inf
    for c in np.arange(1., (ny) + 1):
        M[:, int(c) - 1] = M[:, int(c) - 1] - x + y[int(c) - 1]
        minval = matcompat.max(minval, matcompat.max(M[:, int(c) - 1]))

    [rIdx, cIdx] = nonzero((M == minval))
    return [minval, rIdx, cIdx]
