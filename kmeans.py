import numpy
from matplotlib import pyplot

def runKmeans(X, K, MaxIter):
    """Run K-means clustering algorithm

    X: data matrix where each row is a single data example, each column
      is one dimension or 'feature'
    K: number of clusters to find
    MaxIter: maximum number of iterations
    """
    # Get standard deviation of data points along each dimension. This
    # is used for the stopping criterion.
    std_dev = numpy.std(X, axis=0)
    myrange = numpy.max(X,axis=0) - numpy.min(X,axis=0)
    
    centroids = initialize_centroids(X, K)
    init_centroids = centroids
    old_centroids = centroids

    for ii in range(MaxIter):
        print "iter", ii
        # For each example in X, find the nearest centroid
        idx = findClosestCentroid(X, centroids)
        
        old_centroids = centroids 

        # Given memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

        # Determine if we should stop (because we're almost converged)
        delta = 0.001
        diff = centroids - old_centroids
        print "maxdiff", numpy.max(numpy.abs(diff)), numpy.max(myrange)
##        if numpy.sum(numpy.abs(diff) < delta * myrange) == len(diff):
        
#         if numpy.max(numpy.abs(diff)) == 0.0:
#             return diff, myrange

        # If in all cases the difference is very small, then we quit
        if False not in (numpy.abs(diff) <= delta * myrange):
            print "Stopping at iteration", ii
            break
            

    return centroids, idx


def initialize_centroids(X, K):
    """ Randomly initialize the centroids

    Randomly choose K data points, and assign their positions as
    the centroids
    """
    # Generate array of indices running from 0 to n_examples
    m_examples = len(X[:,0])   # Number of data points
    allind = numpy.arange(m_examples)

    # Randomly shuffle the array of indices
    numpy.random.shuffle(allind)

    # Take only the first K indices
    k_ind = allind[0:K]

    # Select the K points from the data array to be the centroids
    centroids = X[k_ind, :]

    return centroids

def findClosestCentroid(X, centroids):
    """For each example in X, find the nearest centroid"""
    m_examples = len(X[:,0])

    # Initialize array to hold the ID's of each data example's nearest
    # centroid.
    idx = numpy.zeros(m_examples) - 1.0

    # Loop over all data examples, finding closest centroid for each.
    for ii in range(m_examples):
        xx = X[ii,:]

        # Calculate distance between this data example and all 
        # of the centroids
#         if centroids is None:
#             print "STOP!"
#         print centroids, xx

        dist2 = numpy.sum( numpy.power(xx - centroids, 2.0), 1 )

        # Find the index of the centroid that is minimum distance away
        imin = numpy.argmin(dist2)
        idx[ii] = imin

    return idx

def computeCentroids(X, idx, K):
    """Given memberships, find centroid of each cluster"""

    ndim = len(X[0,:])
    centroids = numpy.zeros((K, ndim), dtype='float')

    # Loop over centroids
    for ii in range(K):
        # Find points assigned to this centroid
        wi = (idx == ii)
        xx = X[wi, :]
        
        # Number of data points assigned to this centroid
        npoints_i = len(xx[:,0])

        # Calculate centroid
        centroids[ii,:] = 1.0 / float(npoints_i) * numpy.sum(xx,0)

    return centroids

def test():
    # Create a bunch of data points in (x,y) space
    npoints1 = 30; npoints2 = 22; npoints3 = 15
    x1 = numpy.random.randn(npoints1, 2) * 0.5 + [5, 3.0]
    x2 = numpy.random.randn(npoints2, 2) * 0.7 + [4., 1.0]
    x3 = numpy.random.randn(npoints3, 2) * 0.6 + [1.0, 2.0]

    X = numpy.vstack((x1,x2,x3))

    pyplot.plot(X[:,0], X[:,1], 'o')
    pyplot.show()

    K = 3
    centroids, idx = runKmeans(X, K, 100)

    print 'IDX', numpy.min(idx), numpy.max(idx)
    print "centroids", centroids
    w1 = idx == 0
    w2 = idx == 1
    w3 = idx == 2
    
    print len(w1.nonzero()[0]), len(w2.nonzero()[0]), len(w3.nonzero()[0])
    pyplot.plot(X[w1,0], X[w1,1], 'ob')
    pyplot.plot(X[w2,0], X[w2,1], '+g')
    pyplot.plot(X[w3,0], X[w3,1], 'xb')
    pyplot.show()
    return w1
