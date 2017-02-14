import numpy as np


def munkresFunc(cost_matrix):
    assignment = np.zeros((1, costMat.shape[0]))
    cost = 0.



    # STEP 1
    minR = dMat.min(axis=1)
    print("minR", minR)
    temp_matrix = (dMat.T - minR).T
    print("temp_matrix\n", temp_matrix)
    minC = temp_matrix.min(axis=0)
    print("minC", minC)

    # STEP 2
    zP = dMat == temp_matrix
    print("zP\n", zP)

    starZ = np.zeros(n)
    while np.any(zP.flatten()):
        print("This loop is not tested!")
        r, c = np.nonzero(zP)
        starZ[r] = c
        zP[r, :] = False
        zP[:, c] = False
    print("starZ", starZ)

    while True:
        # STEP 3
        if np.all((starZ > 0.)):
            print("Breaking")
            break

        coverColumn = np.zeros(n, dtype=np.bool)
        coverColumn[starZ > 0.] = True
        print("coverColumn", coverColumn)
        coverRow = np.zeros(n, dtype=np.bool)
        print("coverRow", coverRow)
        primeZ = np.zeros(n)
        print("primeZ", primeZ)

        temp2_0 = dMat[np.ix_(np.logical_not(coverRow), np.logical_not(coverColumn))]
        print("temp2_0\n", temp2_0)
        temp2_1 = minR[np.logical_not(coverRow)].reshape(nRows, 1) + minC[np.logical_not(coverColumn)].reshape(1, nCols)
        print("temp2_1\n", temp2_1)
        temp2 = (temp2_0 == temp2_1)
        print("temp2\n", temp2)
        rIdx, cIdx = np.nonzero(temp2)
        print("rIdx, cIdx", rIdx, cIdx)
        while True:
            cR = np.nonzero(np.logical_not(coverRow))[0]
            print("cR", cR)
            cC = np.nonzero(np.logical_not(coverColumn))[0]
            print("cC",cC)
            rIdx = cR[rIdx]
            cIdx = cC[cIdx]
            Step = 6
            while cIdx.size:
                # STEP 4
                uZr = rIdx[0]
                print("uZr", uZr)
                uZc = cIdx[0]
                print("uZc", uZc)
                primeZ[uZr] = uZc
                print("primeZ",primeZ)
                stz = starZ[uZr]
                print("stz",stz)
                if not stz:
                    Step = 5
                    print("Breaking. Step = 5")
                    break
                print("This code is not tested")
                coverRow[uZr] = True
                coverColumn[stz] = False
                z = rIdx == uZr
                rIdx[z] = np.array([])
                cIdx[z] = np.array([])
                cR = np.nonzero(np.logical_not(coverRow))
                z = dMat[np.logical_not(coverRow), stz] == minR[np.logical_not(coverRow)] + minC[stz]
                # rIdx = np.array(np.vstack((np.hstack((rIdx.flatten(1))), np.hstack((cR[z])))))
                # cIdx = np.array(np.vstack((np.hstack((cIdx.flatten(1))), np.hstack((stz[np.ones(np.sum(z), 1.))])))))

            if Step == 6:
                # STEP 6
                print("This code is not tested")
                [minval, rIdx, cIdx] = outerplus(dMat[int((not coverRow)) - 1, int((not coverColumn)) - 1],
                minR[int((not coverRow)) - 1], minC[int((not coverColumn)) - 1])
                minC[int((not coverColumn)) - 1] = minC[int((not coverColumn)) - 1] + minval
                minR[int(coverRow) - 1] = minR[int(coverRow) - 1] - minval
            else:
                print("Breaking. Step != 6")
                break
        # STEP 5

        rowZ1 = np.nonzero(starZ == uZc)[0]
        print("rowZ1", rowZ1)
        starZ[uZr] = uZc
        print("starZ",starZ)
        while rowZ1 > 0.:
            starZ[rowZ1] = 0.
            uZc = primeZ[rowZ1]
            uZr = rowZ1
            rowZ1 = np.nonzero(starZ == uZc)[0]
            starZ[uZr] = uZc

    rowIdx = np.nonzero(validRow)
    colIdx = np.nonzero(validCol)
    starZ = starZ[rows]
    vIdx = starZ <= nCols
    assignment[rowIdx[vIdx]] = colIdx[starZ[vIdx]]
    pass_vect = assignment[(assignment > 0.)]
    pass_vect[np.logical_not(np.diag(valid_matrix[(assignment > 0.), pass_vect]))] = 0.
    assignment[(assignment > 0.)] = pass_vect
    cost = np.trace(costMat[(assignment > 0.),assignment[(assignment > 0.)]])
    return assignment, cost


def outerplus(M, x, y):
    ny = M.shape[2]
    minval = np.inf
    for c in np.arange(1., (ny) + 1):
        M[:, int(c) - 1] = M[:, int(c) - 1] - x + y[int(c) - 1]
        minval = max(minval, max(M[:, int(c) - 1]))

    [rIdx, cIdx] = nonzero((M == minval))
    return [minval, rIdx, cIdx]


if __name__ == "__main__":
    cost_matrix = np.array(
        [[3, 1, float('inf')],
         [float('inf'), float('inf'), 5],
         [float('inf'), float('inf'), 0.5]])
    print("cost_matrix\n", cost_matrix)

    # Pre-processing
    valid_matrix = cost_matrix < float('inf')
    print("Valid matrix\n", valid_matrix.astype(int))
    bigM = np.power(10., np.ceil(np.log10(np.sum(cost_matrix[valid_matrix]))) + 1.)
    cost_matrix[np.logical_not(valid_matrix)] = bigM
    print("Modified cost matrix\n", cost_matrix)

    validCol = np.any(valid_matrix, axis=0)
    validRow = np.any(valid_matrix, axis=1)
    print("validCol", validCol)
    print("validRow", validRow)
    nRows = int(np.sum(validRow))
    nCols = int(np.sum(validCol))
    n = max(nRows, nCols)
    print("nRows, nCols, n", nRows, nCols, n)

    maxv = 10. * np.max(cost_matrix[valid_matrix])
    print("maxv", maxv)

    rows = np.arange(nRows)
    cols = np.arange(nCols)
    dMat = np.zeros((n, n)) + maxv
    dMat[np.ix_(rows, cols)] = cost_matrix[np.ix_(validRow, validCol)]
    print("dMat\n", dMat)

    # Assignment
    from munkres import Munkres
    m = Munkres()
    preliminary_assignments = m.compute(dMat.tolist())
    print("preliminary assignments", preliminary_assignments)

    # Post-processing
    assignments = []
    for preliminary_assignment in preliminary_assignments:
        row = preliminary_assignment[0]
        col = preliminary_assignment[1]
        if valid_matrix[row,col]:
            assignments.append(preliminary_assignment)
    print("assignments",assignments)