import numpy

#################################### 
#################################### 
def distance(latitude1, longitude1, latitudes, longitudes):
    """Find distance (in meters) between points on Earth's surface
    latitude1 -- single float. latitude of main point
    latitudes -- array-like float.  latitude values of all other points"""

    # Earth radius
    R_earth = 6371e3  # in meters

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = numpy.pi/180.0
         
    # phi = 90 - latitude for Cook's formula
#     phi1 = (90.0 - latitude1)*degrees_to_radians
#     phi2 = (90.0 - latitudes)*degrees_to_radians
    # NB: we use numpy.array for the latitudes to get rid of the table metadata,
    # and just have a normal array
    phi1 = latitude1 * degrees_to_radians
    phi2 = numpy.array(latitudes) * degrees_to_radians

    # theta = longitude
    theta1 = longitude1*degrees_to_radians
    theta2 = numpy.array(longitudes)*degrees_to_radians

    # Haversine formula
    dtheta = theta2 - theta1
    dphi = phi2 - phi1
    aa = (numpy.sin(dphi/2.0))**2.0 + \
        numpy.cos(phi1) * numpy.cos(phi2) * numpy.sin(dtheta/2.0)**2.0
    arc = 2 * numpy.arctan2(aa**0.5, (1 - aa)**0.5)
    dist = arc * R_earth

    # Following from John D. Cook website.
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

#     cos = (numpy.sin(phi1)*numpy.sin(phi2)*numpy.cos(theta1 - theta2) + 
#            numpy.cos(phi1)*numpy.cos(phi2))
#     arc = numpy.arccos( cos )

#     dist = arc * R_earth

    # For speed, we can simplify the above by assuming that the points
    # are close together on Earth's surface, so the angle differences
    # are close to 0 degrees.

    return dist
 
############################ 
############################ 
def find_nearby_stations(lat1, long1, latitudes, longitudes, maxdis=300,
                         n_station = 0):
    """Find bike stations near a point on Earth's surface.
    Returns an index array to the latitudes and longitudes that are close.
    If no bike stations are close enough, then return the index of the 
    closest station.
    maxdis is the search radius for finding stations.  If maxdis <= 0, then
    return the n_station nearest stations."""
    
    dis = distance(lat1, long1, latitudes, longitudes)

    # find distances less than maxdis
    w_select = (dis <= maxdis)

    # If no stations are within maxdis, just find the nearest station
    if numpy.sum(w_select) < 1:
        w_select = numpy.array([numpy.argmin(dis)])

    else:
        # convert from an n_latitudes array of True and False values to
        # an N_match array array indices
        w_select = (numpy.nonzero(w_select))[0]

        # sort the stations from nearest to farthest
        ss = numpy.argsort(dis[w_select])
        w_select = w_select[ss]


    dis_select = dis[w_select]

    # also return the distances of the resulting stations?
    return w_select, dis_select


############################ 
############################ 
def normalize_timeseries(timeseries, maxval=None, minval=None):
    """Normalize a time series so that its values fall between 0.0 and 1.0"""
    if maxval is None:
        maxval = numpy.max(timeseries)
    if minval is None:
        minval = numpy.min(timeseries)

    norm_timeseries = timeseries.copy() - minval
    norm_timeseries /= (maxval - minval)

    return norm_timeseries
    


############################ 
############################ 
def test(dat):
    
    ## Test the distance measures (compare to e.g. Movable Type Scripts website)
    all_lat = dat['lat']
    all_long = dat['long']
    
    ind = 50
    mylat = all_lat[ind]
    mylong = all_long[ind]

    w_near, dis = find_nearby_stations(mylat, mylong, all_lat, all_long, maxdis=500)

    print len(w_near)
    print w_near
    print mylat, mylong
    print dis
##    print all_lat[w_near], all_long[w_near], dis
    


