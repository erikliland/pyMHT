import numpy as np
from timeit import default_timer as timer
from numba import vectorize

def pyVectorAdd(a,b,c):
    for i in range(a.size):
        c[i] = a[i] + b[i]

def numpyVectorAdd(a,b):
    return a+b

@vectorize(["float32(float32, float32)"], target='cpu')
def cpuVectorAdd(a,b):
    return a + b


@vectorize(["float32(float32, float32)"], target='parallel')
def parallelVectorAdd(a,b):
    return a + b


@vectorize(["float32(float32, float32)"], target='cuda')
def cudaVectorAdd(a,b):
    return a + b


def main():
    N = 3200

    A = np.ones(N,dtype=np.float32)
    B = np.zeros(N, dtype=np.float32)
    C1 = np.zeros(N,dtype=np.float32)

    start0 = timer()
    C0 = numpyVectorAdd(A, B)
    time0 = timer() - start0

    start1 = timer()
    pyVectorAdd(A,B, C1)
    time1 = timer()-start1

    start2 = timer()
    C2 = cpuVectorAdd(A,B)
    time2 = timer()-start2

    start3 = timer()
    C3 = parallelVectorAdd(A,B)
    time3 = timer()-start3

    start4 = timer()
    C4 = cudaVectorAdd(A,B)
    time4 = timer()-start4

    # print("C_RADAR[:5] = " + str(C_RADAR[:5]))
    # print("C_RADAR[-5:] = " + str(C_RADAR[-5:]))

    print("VectorAdd took", time0*1000, time1*1000, time2*1000,time3*1000, time4*1000, "ms", sep = "\n")


def main2():
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=9.5, llcrnrlat=63.2,
                urcrnrlon=10.9, urcrnrlat=64.,
                resolution='i', projection='tmerc',
                lon_0=10.7, lat_0=63.4)
    # can get the identical map this way (by specifying width and
    # height instead of lat/lon corners)
    # m = Basemap(width=894887,height=1116766,\
    #            resolution='i',projection='tmerc',lon_0=-4.36,lat_0=54.7)
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    # m.drawparallels(np.arange(-40, 61., 2.))
    # m.drawmeridians(np.arange(-20., 21., 2.))
    m.drawmapboundary(fill_color='aqua')
    plt.title("Transverse Mercator Projection")
    plt.show()

if __name__ == "__main__":
    main2()