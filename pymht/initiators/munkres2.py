
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def munkres(costMat):

    # Local Variables: coverColumn, validRow, coverRow, colIdx, pass, cost, bigM, costMat, uZc, rowZ1, starZ, nCols, minval, cC, minC, maxv, nRows, Step, primeZ, validMat, cR, minR, zP, validCol, assignment, vIdx, rIdx, c, rowIdx, n, cIdx, r, stz, dMat, z, uZr
    # Function calls: true, outerplus, all, false, trace, min, munkres, max, sum, bsxfun, ceil, find, ones, zeros, diag, isempty, Inf, log10, any, size
    assignment = np.zeros(1., matcompat.size(costMat, 1.))
    cost = 0.
    validMat = np.logical_and(costMat == costMat, costMat<Inf)
    bigM = 10.**(np.ceil(np.log10(np.sum(costMat[int(validMat)-1])))+1.)
    costMat[int((not validMat))-1] = bigM
    validCol = np.any(validMat, 1.)
    validRow = np.any(validMat, 2.)
    nRows = np.sum(validRow)
    nCols = np.sum(validCol)
    n = matcompat.max(nRows, nCols)
    maxv = 10.*matcompat.max(costMat[int(validMat)-1])
    dMat = np.zeros(n)+maxv
    dMat[0:nRows,0:nCols] = costMat[int(validRow)-1,int(validCol)-1]
    minR = matcompat.max(dMat, np.array([]), 2.)
    minC = matcompat.max(bsxfun(minus, dMat, minR))
    zP = dMat == bsxfun(plus, minC, minR)
    starZ = np.zeros(n, 1.)
    while np.any(zP.flatten(1)):
        [r, c] = nonzero(zP, 1.)
        starZ[int(r)-1] = c
        zP[int(r)-1,:] = false
        zP[:,int(c)-1] = false
        
    while 1.:
        if np.all((starZ > 0.)):
            break
        
        
        coverColumn = false(1., n)
        coverColumn[int(starZ[int((starZ > 0.))-1])-1] = true
        coverRow = false(n, 1.)
        primeZ = np.zeros(n, 1.)
        [rIdx, cIdx] = nonzero((dMat[int((not coverRow))-1,int((not coverColumn))-1] == bsxfun(plus, minR[int((not coverRow))-1], minC[int((not coverColumn))-1])))
        while 1.:
            cR = nonzero((not coverRow))
            cC = nonzero((not coverColumn))
            rIdx = cR[int(rIdx)-1]
            cIdx = cC[int(cIdx)-1]
            Step = 6.
            while not isempty(cIdx):
                uZr = rIdx[0]
                uZc = cIdx[0]
                primeZ[int(uZr)-1] = uZc
                stz = starZ[int(uZr)-1]
                if not stz:
                    Step = 5.
                    break
                
                
                coverRow[int(uZr)-1] = true
                coverColumn[int(stz)-1] = false
                z = rIdx == uZr
                rIdx[int(z)-1] = np.array([])
                cIdx[int(z)-1] = np.array([])
                cR = nonzero((not coverRow))
                z = dMat[int((not coverRow))-1,int(stz)-1] == minR[int((not coverRow))-1]+minC[int(stz)-1]
                rIdx = np.array(np.vstack((np.hstack((rIdx.flatten(1))), np.hstack((cR[int(z)-1])))))
                cIdx = np.array(np.vstack((np.hstack((cIdx.flatten(1))), np.hstack((stz[int(np.ones(np.sum(z), 1.))-1])))))
                
            if Step == 6.:
                [minval, rIdx, cIdx] = outerplus(dMat[int((not coverRow))-1,int((not coverColumn))-1], minR[int((not coverRow))-1], minC[int((not coverColumn))-1])
                minC[int((not coverColumn))-1] = minC[int((not coverColumn))-1]+minval
                minR[int(coverRow)-1] = minR[int(coverRow)-1]-minval
            else:
                break
                
            
            
        rowZ1 = nonzero((starZ == uZc))
        starZ[int(uZr)-1] = uZc
        while rowZ1 > 0.:
            starZ[int(rowZ1)-1] = 0.
            uZc = primeZ[int(rowZ1)-1]
            uZr = rowZ1
            rowZ1 = nonzero((starZ == uZc))
            starZ[int(uZr)-1] = uZc
            
        
    rowIdx = nonzero(validRow)
    colIdx = nonzero(validCol)
    starZ = starZ[0:nRows]
    vIdx = starZ<=nCols
    assignment[int(rowIdx[int(vIdx)-1])-1] = colIdx[int(starZ[int(vIdx)-1])-1]
    pass = assignment[int((assignment > 0.))-1]
    pass[int((not np.diag[int(validMat[int((assignment > 0.))-1,int(pass)-1])-1]))-1] = 0.
    assignment[int((assignment > 0.))-1] = pass
    cost = np.trace(costMat[int((assignment > 0.))-1,int(assignment[int((assignment > 0.))-1])-1])
    return [assignment, cost]
def outerplus(M, x, y):

    # Local Variables: c, minval, M, cIdx, ny, rIdx, y, x
    # Function calls: inf, min, outerplus, find, size
    ny = matcompat.size(M, 2.)
    minval = np.inf
    for c in np.arange(1., (ny)+1):
        M[:,int(c)-1] = M[:,int(c)-1]-x+y[int(c)-1]
        minval = matcompat.max(minval, matcompat.max(M[:,int(c)-1]))
        
    [rIdx, cIdx] = nonzero((M == minval))
    return [minval, rIdx, cIdx]