#
# data in ~/Documents/Junk/VELIB/SF_bikeshare
#
import numpy
import cPickle as pickle
import matplotlib.pyplot as pyplot
import os

import astropy 
from astropy.io import ascii

from scipy.stats import binned_statistic

# import tables
# import time
import datetime

def read_rebalance(file = '201408_rebalancing_data.csv', unpickle=0):
    """Read SF bikeshare rebalancing data"""

    if unpickle > 0:
        file = 'sfdat.pkl'
        ff = open(file, 'rb')
        data_table = pickle.load(ff)
        ff.close()
        return data_table

    # read the original ascii table
    converters = {'station_id':[astropy.io.ascii.convert_numpy(numpy.int8)],
                  'bikes_available':
                      [astropy.io.ascii.convert_numpy(numpy.int8)],
                  'docks_available':
                      [astropy.io.ascii.convert_numpy(numpy.int8)],
                  'time':[astropy.io.ascii.convert_numpy(numpy.str)]}

    data_table = ascii.read(file, format='csv', 
##                            data_start = 1, data_end = 2000, 
                            converters=converters, fast_reader =True)

    return data_table


def read_station(file = '201408_station_data.csv'):
    """Read SF bikeshare station data (locations)"""
    converters = {'station_id':[astropy.io.ascii.convert_numpy(numpy.int8)],
                  'name':[astropy.io.ascii.convert_numpy(numpy.str)],
                  'lat':[astropy.io.ascii.convert_numpy(numpy.float64)],
                  'long':[astropy.io.ascii.convert_numpy(numpy.float64)],
                  'dockcount':[astropy.io.ascii.convert_numpy(numpy.int8)],
                  'landmark':[astropy.io.ascii.convert_numpy(numpy.str)],
                  'installation':[astropy.io.ascii.convert_numpy(numpy.str)],
                  'notes':[astropy.io.ascii.convert_numpy(numpy.str)]}
    data_table = ascii.read(file, format='csv',
                            converters=converters, fast_reader=True)
    return data_table

###def fold_weekly(data):
def adjust_times(data):
    """Convert the time strings into datetime64 objects, and fold data
    to get the minute-of-week time."""
    minutes_per_day = 24 * 60
    format = '%Y-%m-%d %H:%M:%S'   # for time string

    # Define the time zone.  Default is Pacific time.
    timezone = USTimeZone() 

    dd = data

    t_norm = numpy.zeros(len(dd))
    t_dt64 = numpy.zeros(len(dd),'datetime64[s]')
    i = 0
    for timestring in dd['time']:
        # make a datetime object from the original time string
        dt = datetime.datetime.strptime(timestring, format)

        # Add time zone information
        dt = dt.replace(tzinfo=timezone)

        # get the day of week 
        dow = dt.strftime('%w')
        
        # Convert to the time, in minutes, from the start of the week
        t_norm[i] = (float(dow) * minutes_per_day) + \
            dt.hour * 60.0 + float(dt.minute) + dt.second / 60.

        # Convert the python datetime object to a numpy.datetime64 object
        # The 's' is to force the units to be seconds
        t_dt64[i] = numpy.datetime64(dt.isoformat(), 's')

###        print t_dt64[i]

        i += 1

    # Add t_norm column to the table.  t_norm is the time, in minutes, from the 
    # start of the week.
    dd['t_norm'] = t_norm.astype("float32")

    # Swap the time string for a datetime64 object
    # First remove the original time column from the data table
    dd.remove_column('time')

    # Now add a new time column where the data values are datetime64 objects
    dd['time'] = t_dt64
    
    return dd

######################
def test():
    dat = read_rebalance(unpickle=1)
    print "done reading data"
##    dd = fold_weekly(dat)
##    print "done getting weekly normalization... now saving"

    # save the data
#     file = open('sfdat.pkl', 'wb')
#     pickle.dump(dd, file, -1)
#     file.close
#     print "Done saving"
    
    myID = 68
    plot_weekly(dat, myID)
    
    return dat
##    return dd


def plot_weekly(dat, station_id):
    """Plot weekly-folded data"""
    w_id = dat['station_id'] == station_id
    xx = dat[w_id]['t_norm'] / 60
    yy = dat[w_id]['bikes_available']
    pyplot.plot(xx, yy, '+y')

    median, bin_edges, binnum = \
        binned_statistic(xx, yy, statistic='median', bins=1000)
    pyplot.plot(bin_edges[1:], median)
    pyplot.xlabel("Time of week (hours)")
    pyplot.ylabel("bikes available")
    
    pyplot.show()

    
def plot_section(dat, station_id):
    dd = dat[dat['station_id'] == station_id]
    yy = dd['bikes_available']
##    xx = dd['time']
    
    npoints = 360
    firstp = 5000
    w_select = numpy.arange(npoints) + firstp
    pyplot.plot(yy[w_select], '+y')
    pyplot.show()
    


######################################
######################################
# Following defines a tzinfo (time zone info) class
# which allows e.g. to convert easily between times
# given in the original data files and UTC time.
# Adapted from the datetime module documentation page.
######################################
ZERO = datetime.timedelta(0)
HOUR = datetime.timedelta(hours=1)
def first_sunday_on_or_after(dt):
    days_to_go = 6 - dt.weekday()
    if days_to_go:
        dt += datetime.timedelta(days_to_go)
    return dt

# US DST Rules
#
# This is a simplified (i.e., wrong for a few cases) set of rules for US
# DST start and end times. For a complete and up-to-date set of DST rules
# and timezone definitions, visit the Olson Database (or try pytz):
# http://www.twinsun.com/tz/tz-link.htm
# http://sourceforge.net/projects/pytz/ (might not be up-to-date)
#
# In the US, since 2007, DST starts at 2am (standard time) on the second
# Sunday in March, which is the first Sunday on or after Mar 8.
DSTSTART_2007 = datetime.datetime(1, 3, 8, 2)
# and ends at 2am (DST time; 1am standard time) on the first Sunday of Nov.
DSTEND_2007 = datetime.datetime(1, 11, 1, 1)
# From 1987 to 2006, DST used to start at 2am (standard time) on the first
# Sunday in April and to end at 2am (DST time; 1am standard time) on the last
# Sunday of October, which is the first Sunday on or after Oct 25.
DSTSTART_1987_2006 = datetime.datetime(1, 4, 1, 2)
DSTEND_1987_2006 = datetime.datetime(1, 10, 25, 1)
# From 1967 to 1986, DST used to start at 2am (standard time) on the last
# Sunday in April (the one on or after April 24) and to end at 2am (DST time;
# 1am standard time) on the last Sunday of October, which is the first Sunday
# on or after Oct 25.
DSTSTART_1967_1986 = datetime.datetime(1, 4, 24, 2)
DSTEND_1967_1986 = DSTEND_1987_2006

class USTimeZone(datetime.tzinfo):

###    def __init__(self, hours, reprname, stdname, dstname):
## JMG: Change definition so that __init__() can be called without
## arguments. The datetime module documentation indicates that this
## is necessary for pickling to work correctly.
    def __init__(self, hours=-8, reprname='Pacific', 
                 stdname='PST', dstname='PDT'):
        self.stdoffset = datetime.timedelta(hours=hours)
        self.reprname = reprname
        self.stdname = stdname
        self.dstname = dstname

    def __repr__(self):
        return self.reprname

    def tzname(self, dt):
        if self.dst(dt):
            return self.dstname
        else:
            return self.stdname

    def utcoffset(self, dt):
        return self.stdoffset + self.dst(dt)

    def dst(self, dt):
        if dt is None or dt.tzinfo is None:
            # An exception may be sensible here, in one or both cases.
            # It depends on how you want to treat them.  The default
            # fromutc() implementation (called by the default astimezone()
            # implementation) passes a datetime with dt.tzinfo is self.
            return ZERO
        assert dt.tzinfo is self

        # Find start and end times for US DST. For years before 1967, return
        # ZERO for no DST.
        if 2006 < dt.year:
            dststart, dstend = DSTSTART_2007, DSTEND_2007
        elif 1986 < dt.year < 2007:
            dststart, dstend = DSTSTART_1987_2006, DSTEND_1987_2006
        elif 1966 < dt.year < 1987:
            dststart, dstend = DSTSTART_1967_1986, DSTEND_1967_1986
        else:
            return ZERO

        start = first_sunday_on_or_after(dststart.replace(year=dt.year))
        end = first_sunday_on_or_after(dstend.replace(year=dt.year))

        # Can't compare naive to aware objects, so strip the timezone from
        # dt first.
        if start <= dt.replace(tzinfo=None) < end:
            return HOUR
        else:
            return ZERO

Eastern  = USTimeZone(-5, "Eastern",  "EST", "EDT")
Central  = USTimeZone(-6, "Central",  "CST", "CDT")
Mountain = USTimeZone(-7, "Mountain", "MST", "MDT")
Pacific  = USTimeZone(-8, "Pacific",  "PST", "PDT")


# A UTC class.
class UTC(datetime.tzinfo):
    """UTC tzinfo class"""

    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO

utc = UTC()
