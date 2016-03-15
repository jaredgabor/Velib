#
# Data in:
#/Users/jgabor/Documents/Junk/VELIB
#
#
import numpy
from numpy import fft as FFT
import cPickle as pickle
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import matplotlib.patches as patches

import os.path
import sys
from re import match as strmatch
from scipy.stats import percentileofscore
from scipy import signal
import statsmodels.tsa.stattools as stats
from scipy.ndimage import filters
from sklearn.svm import SVR


##import astropy
import datetime

import sf_bike
import bike_utils
import weatherdat
reload(weatherdat)

import kmeans
reload(kmeans)

from astropy.table import column as Column
from astropy.io import ascii

reload(sf_bike)
reload(bike_utils)


SECONDS_PER_MINUTE = 60
MINUTES_PER_WEEK = 7*24*60
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = MINUTES_PER_HOUR * 24
####################################
####################################
class PredictionObj:
    def __init__(self, distribution, quantity_name):
        """Object for making predictions of a quantity.
        
        distribution: sample distribution of the quantity
        quantity_name: string; gives the name of the quantity (so far
           only used to label plots)
        """
        self.distribution = distribution
        self.name = quantity_name

    def percentile(self, percentiles):
        """Return desired percentiles of the distribution"""
        return numpy.percentile(self.distribution, percentiles)

    def med(self):
        """Return the median of the distribution"""
        return numpy.median(self.distribution)

    def prediction(self):
        """Return a 'best prediction' of the distribution (e.g. median)"""
        return self.med()

    def prob_atleast(self, number):
        """Return the probability that at least 'number' bikes will be 
        available"""
        return 100.0 - \
            percentileofscore(self.distribution, number, 'weak')
        
    def plot_dist(self):
        """Quickly plot the underlying probability distribution"""
        fig = pyplot.figure()
        pyplot.hist(self.distribution)
        pyplot.xlabel(self.name)
        pyplot.show()


####################################
####################################
class BikeData:
    """Class for reading, manipulating, analyzing bike share data"""

    ####################################
    def __init__(self, unpickle=0, save_pickle=0, 
                 rebalance_data = None,
                 timeline_filename = None, stationlist_filename=None):
        """Read in bike sharing data

        unpickle: This is passed to read_rebalance.  If >0, then we
          will attempt to read from a pickle file (whose name is hard-coded)
          to obtain the rebalancing data.

        rebalance_data: If set, then we will use this as the rebalancing
          data rather than reading the data from a file.

        timeline_filename: Name of a pickle file which stores data for
          the common Timeline object.  If provided, we will read the file
          for the Timeline data rather than using the underlying 
          rebalancing data to construct a timeline from scratch.

        stationlist_filename: Name of a pickle file which stores data for
          many Station objects.  If provided, then we will read the file
          when building the station list rather than creating new station data.
        
        """
        self.setup_vars()

        # Read in the rebalancing data (or just copy it from the 
        # arguments provided by the caller).
        if rebalance_data is None:
            self.rebalancing_data = \
                self.read_rebalance(unpickle=unpickle, 
                                    save=save_pickle)
##            self.check_gaps_all()
        else:
            self.rebalancing_data = rebalance_data

        # Read the station data (e.g. GPS coordiates)
        self.station_data = self.read_station()

        # Cross-match station IDs from rebalance data and station data.
        # Delete any stations that don't appear in both.
        self.cross_match_ids()

#         # Array to hold "cleaned" timeseries (i.e. normalized, filtered)
#         self.clean_ts_arr = self.read_clean_ts(unpickle=unpickle)
#         # List to hold information about each station's neighbors
#         self.neighbor_list = None

        # Save the common timeline for later
        self.setup_timeline(load_filename = timeline_filename)

        # Get weather info
        self.weather = None
        self.read_weather()

        # Create list of Station objects
        print 'Now build station list'
        self.build_station_list(load_filename = stationlist_filename)
        print "done building station list"

    ####################################
    def cross_match_ids(self):
        """Cross-match station IDs from station data and rebalancing data"""
        
        # I had to use numpy.array() here because otherwise I get an error
        # related to the masking properties when calling setxor1d()
        ids_rebal = numpy.array(
            numpy.unique(self.rebalancing_data['station_id']))
        ids_stat = numpy.array(
            numpy.unique(self.station_data['station_id']))
        
##        print ids_rebal, ids_stat

        # Find IDs that are present in only one of the two lists.
        bad_ids = numpy.setxor1d(ids_rebal, ids_stat)

        # Eliminate rows with these bad IDs
        for bad_id in bad_ids:
            # Do the elimination for both tables
            for table in [self.rebalancing_data, self.station_data]:
                # Find rows with the bad ID
                wbad = numpy.where(
                    table['station_id'] == bad_id)[0]
                if len(wbad) > 0:
                    table.remove_rows(wbad)


    ####################################
    def setup_timeline(self, load_filename=None):
        """Setup the Timeline object for this bikedata object

        This sets "self.common_timeline" to a new Timeline object.
        """
        if load_filename is not None:
            self.common_timeline = Timeline(0, load_filename=load_filename)

        else:
            first_station = self.station_data['station_id'][0]
            wid = self.rebalancing_data['station_id'] == \
                first_station
            timezone = sf_bike.USTimeZone()
            self.common_timeline = Timeline(
                numpy.array(self.rebalancing_data['time'][wid]),
                timezone = timezone)

        # Check whether various attributes have been set.  If not
        # then do the calculations necessary.
        if self.common_timeline.datetimes is None:
            print "Making datetimes on common timeline"
            self.common_timeline.make_datetimes()

        if self.common_timeline.ind_daystart is None:
            print "Finding daystarts for common timeline..."
            self.common_timeline.find_all_daystarts()
            self.common_timeline.match_dayends()

        if self.common_timeline.tnorm is None:
            self.common_timeline.tnorm_weekly()


    ####################################
    def read_weather(self):
        """Read weather data
        """
        # Read the weather data from file.  File is assumed to be known.
        self.weather = weatherdat.Weather(
            timezone=self.common_timeline.timezone)
        
        # Make features for regression.
        self.weather.make_weather_features(
            self.common_timeline.datetimes)


    ####################################
    def save_stations(self, filename):
        """Save underlying data for each Station object in this
        bikedata object.
        
        """
        station_list = self.all_stations.station_list

        print "Saving %d stations in file %s" % (len(station_list), filename)
        
        # Open file for writing.
        file = open(filename, 'wb')

        # First save the number of stations
        pickle.dump(len(station_list), file, -1)

        # Loop through stations, saving data for each.
        for station in station_list:
            station.save_data(file)

        # close the pickle file
        file.close()

        pass

    ####################################
    def load_stations(self, filename):
        """Load data from stations that had previously been saved.

        """
        # Open the file for reading.
        file = open(filename, 'rb')
        
        # Read in the number of stations
        n_stations = pickle.load(file)

        print "Loading %d stations from file %s" % (n_stations, filename)

        # Loop through the appropriate number of stations, creating
        # a new Station object for each and reading the underlying
        # data from the file
        station_list = []
        for ii in range(n_stations):
            station = Station(0,0,0, 
                              timeline=self.common_timeline,
                              weather = self.weather,
                              loadfile = file)
            station_list.append(station)

        # Close the pickle file
        file.close()

        self.all_stations = StationGroup(station_list)


    ####################################
    def save_data(self, do_rebalance=False, do_station=False, 
                  do_clean_ts=False, do_timeline = False,
                  do_stationlist = False):
        """ Save specific data components of the bike data structure.
        """

        rebal_filename = self.dir + "sfdat.pkl"
        station_filename = self.dir + 'sf_stations.pkl'
        clean_ts_filename = self.dir + 'clean_ts.pkl'
        timeline_filename = self.dir + 'common_timeline.pkl'
        stationlist_filename = self.dir + 'stationlist.pkl'

        # Save the database in a pickle file
        if do_rebalance:
            print "Saving rebalancing_data in file "+rebal_filename
            file = open(rebal_filename, 'wb')
            pickle.dump(self.rebalancing_data,file,-1)
            file.close()

        if do_station:
            print "Saving station_data in file "+station_filename
            file = open(station_filename, 'wb')
            pickle.dump(self.station_data, -1)
            file.close()

        # Save the 'clean' time series array.
        if do_clean_ts:
            if self.clean_ts_arr is None:
                print "Cannot save empty clean_ts array.  Ignoring."
            else:
                print "Saving cleaned time series data in file" + \
                    clean_ts_filename
##                file = open(clean_ts_filename,'wb')
                (self.clean_ts_arr).dump(clean_ts_filename)
##                file.close()

        if do_timeline:
            self.common_timeline.save_data(timeline_filename)

        if do_stationlist:
            self.save_stations(stationlist_filename)

        
    ####################################
    def setup_vars(self):
        """This should be redefined by subclasses to specify filenames 
        used for the 'read' routines"""
        pass

    ####################################
    def read_rebalance(self):
        """This should be redefined by specific subclasses"""
        return 0

    ####################################
    def read_station(self):
        """This should be redefined by specific subclasses"""
        return 0

    ####################################
    def read_clean_ts(self):
        """This should be redefined by specific subclasses"""
        return 0

    ####################################
    def predict_simple(self, station_id,
                       timestring, fmt='%Y-%m-%d %H:%M:%S',
                       quantity = 'bikes_available'):
        """Use simple extrapolation of weekly-folded bike data to estimate 
        future bike (or stand) availability at a given time specified by 
        'timestring'.

        timestring: a string indicating the time for which a prediction should
            be made.
        fmt: the format of the timestring
        quantity: 'bikes_available' (default) or 'docks_available'
        """

        # select adjacent times within this many minutes
        time_window_width = 5.0
    
        # Get just the station of interest
        dd = data[data['station_id'] == station_id]

        # Convert timestring to time of week
        TOW = time_of_week(timestring, fmt=fmt)
    
        # Select the data at the desired time.
        w_select = (dd["t_norm"] > TOW - time_window_width) * \
            (dd['t_norm'] < TOW + time_window_width)

        # Return a Prediction object
        return PredictionObj(dd[w_select][quantity], quantity)

    ####################################
    def predict_simple2(self, station_id,
                       timestring, fmt='%Y-%m-%d %H:%M:%S',
                       quantity = 'bikes_available', tnorm=None):
        """Use simple extrapolation of weekly-folded bike data to estimate 
        future bike (or stand) availability at a given time specified by 
        'timestring'.

        timestring: a string indicating the time for which a prediction should
            be made.
        fmt: the format of the timestring
        quantity: 'bikes_available' (default) or 'docks_available'
        """

        # select adjacent times within this many minutes
        time_window_width = 5.0
    
        # Get just the station of interest
        mystation = self.all_stations.get_station(station_id)

        # Convert user-provided timestring to time of week
        TOW = time_of_week(timestring, fmt=fmt)

        print "weekday:", TOW/1440

        # Find tnorm for the timeline -- the time normalized so that
        # it gives the minute of the week
        if tnorm is None:
            print "get tnorm"
            tnorm = mystation.timeline.tnorm_weekly()
            print "done tnorm"
            pass

        # Select the data at the desired time.
        w_select = (tnorm > (TOW - time_window_width)) * \
            (tnorm < (TOW + time_window_width))

        ##### Plot consecutive values from a single tnorm. DEBUG
        w_select = (tnorm >= (TOW)) * \
            (tnorm < (TOW +1))

        print 'N select', len(w_select.nonzero())
        
        # Show a time-series view 
        xx = tnorm[w_select]
        ts = mystation.table[w_select][quantity]
        pyplot.plot(ts, 'o')
        pyplot.show()
        ###### End debug stuff

        # Return a Prediction object
        return PredictionObj(mystation.table[w_select][quantity], 
                             quantity)

    ####################################
    def predict_simple3(self, station_id,
                       timestring, fmt='%Y-%m-%d %H:%M:%S',
                       quantity = 'bikes_available', tnorm=None):
        """Use simple extrapolation of DAILY-folded bike data to estimate 
        future bike (or stand) availability at a given time specified by 
        'timestring'.

        timestring: a string indicating the time for which a prediction should
            be made.
        fmt: the format of the timestring
        quantity: 'bikes_available' (default) or 'docks_available'
        """

        # select adjacent times within this many minutes
        time_window_width = 5.0
    
        # Get just the station of interest
        mystation = self.all_stations.get_station(station_id)

        # Convert user-provided timestring to time of week
        TOW = time_of_day(timestring, fmt=fmt)

        # Find tnorm for the timeline -- the time normalized so that
        # it gives the minute of the week
        if tnorm is None:
            print "get tnorm"
            tnorm = mystation.timeline.tnorm_daily()
            print "done tnorm"
            pass

        # Select the data at the desired time.
        w_select = (tnorm > (TOW - time_window_width)) * \
            (tnorm < (TOW + time_window_width))

        # Return a Prediction object
        return PredictionObj(mystation.table[w_select][quantity], 
                             quantity)

    ####################################
    def plot_full_series(self, station_id=68,
                         quantity='bikes_available'):
        """Plot full time series data for one station"""
        seconds_per_day = 3600 * 24

        w_id = self.rebalancing_data['station_id'] == station_id

        # x-data is time in days
        xdat = (self.rebalancing_data[w_id]['time']).astype('int') / \
            float(seconds_per_day)
        xdat = xdat - numpy.min(xdat)

        # y-data is bikes available or docks available
        ydat = (self.rebalancing_data[w_id][quantity])

        pyplot.plot(xdat,ydat)
##        pyplot.plot(ydat)
        pyplot.xlabel("Time (days)")
        pyplot.ylabel(quantity)

        pyplot.show()

        pass


    ####################################
    def plot_derivative(self, station_id=68,
                        quantity='bikes_available'):
        """Plot derivative of time series data"""
        seconds_per_day = 3600 * 24

        w_id = self.rebalancing_data['station_id'] == station_id

        # x-data is time in days
        xdat = (self.rebalancing_data[w_id]['time']).astype('int') / \
            float(seconds_per_day)
        xdat = xdat - numpy.min(xdat)

        # y-data is bikes available or docks available
        ydat = (self.rebalancing_data[w_id][quantity])

        # y-data should be time derivative of 'quantity'
        dt = 1.0  # arbitrary. ==constant means we assume all dt are 1 minute
        ydat = (numpy.roll(ydat) - ydat) / dt


        pyplot.plot(xdat,ydat)
##        pyplot.plot(ydat)
        pyplot.xlabel("Time (days)")
        pyplot.ylabel('d(' + quantity + ')/dt')

        pyplot.show()

        pass

    ####################################
    def plot_fourier(self, station_id, power=True):
        """Plot fourier transform for one station's bike availability

        power: if True, plot the Fourier power spectrum
        """
        myfft = self.one_fourier(station_id, power=power, shift=True)

        # NB: myfft may include complex values, in which case this is the
        # complex logarithm function.  The complex logarithm of 
        # x = r*e^(i*theta) is log(x) = log(r) + i*theta.  When we plot later
        # pyplot only shows the real part, which for the log is log(r), which
        # is the logarithm of the magnitude of x.  So by taking the logarithm 
        # now and plotting later, we are plotting the logarithm of the
        # magnitude of the fft (I think this is also the square root of the
        # power spectrum).
        ydat = numpy.log10((myfft))

###        print ydat[0:10]

        # Should we plot the power spectrum or just the FFT?
        if power:
            mylabel = 'Log(Power spectrum)'
        else:
            mylabel = 'Log(Fourier transform)'

        npoints = len(myfft)

        # Plot scale (instead of frequency) on x-axis?
        xdat = numpy.arange(npoints) - npoints/2.0  # Frequency
##        xdat = npoints / xdat  / 60    ##3600.  # Scale

        pyplot.plot(xdat, ydat)
        pyplot.ylabel(mylabel)

##        print xdat*1e4
        pyplot.show()
        
        pass

    ####################################
    def bad_bikes(self, station_id):
        """Identify bad bikes from patterns in the bike availability data"""
        
        pass

    
    
    ####################################
    def check_gaps_all(self):
        """Run check_gaps on all stations"""
        print "Checking gaps"
        for station_id in numpy.unique(self.rebalancing_data['station_id']):
##            sys.stdout.write(str(station_id))
            print station_id,
            sys.stdout.flush()
            self.check_gaps(station_id)

#            w_id = self.rebalancing_data['station_id'] == station_id
#            print len(self.rebalancing_data[w_id])
            

    ####################################
    def check_gaps(self, station_id):
        """Check for gaps in the time series data (e.g. more than
        one minute between points)"""
        # Find this station's data
        w_id = self.rebalancing_data['station_id'] == station_id
        data_this_station = self.rebalancing_data[w_id]

        times = (data_this_station['time']).astype('int') / \
            SECONDS_PER_MINUTE
            
        # Make sure this station has the same starting and stopping
        # times as other stations. (We will later fill in any gaps this
        # creates.)
        maxtime = numpy.max(self.rebalancing_data['time']).astype('int') / \
            SECONDS_PER_MINUTE
        mintime = numpy.min(self.rebalancing_data['time']).astype('int') / \
            SECONDS_PER_MINUTE
        if numpy.min(times) > mintime:
            # Add a row with this minimum time at the beginning
            insert_index = (numpy.where(w_id)[0])[0]
            print "INSERT_INDEX:", insert_index
            self.rebalancing_data.insert_row(
                insert_index,
                {'station_id':station_id, 'bikes_available':-1,
                 'docks_available':-1, 
                 'time': numpy.datetime64(mintime * SECONDS_PER_MINUTE,'s'),
                 't_norm':-1})
            pass
        if numpy.max(times) < maxtime:
            # Add a row with this maximum time at the end
###            n_this_station = len(data_this_station)
            insert_index = (numpy.where(w_id)[0])[-1] + 1
            print "INSERT_INDEX:", insert_index
            self.rebalancing_data.insert_row(
                insert_index,
                {'station_id':station_id, 'bikes_available':-1,
                 'docks_available':-1, 
                 'time': numpy.datetime64(maxtime * SECONDS_PER_MINUTE, 's'),
                 't_norm':-1})
            pass

        # The data table has been modified, so we need to redefine
        # which rows correspond to this station
        w_id = self.rebalancing_data['station_id'] == station_id
        data_this_station = (self.rebalancing_data[w_id]).copy()

        # Get this station's times, in minutes
        # Should we round here?
        times = (data_this_station['time']).astype('int') / \
            SECONDS_PER_MINUTE

        # Ensure that the times increase by 1 minute each
        timeshift = numpy.roll(times,-1)
        diff = timeshift - times
        
        # Where are there gaps in the times?
        wgap = (numpy.where(diff > 1))[0]

        # At locations of gaps, add new time points and assume 
        # bikes_available and docks_available remain the same.
        total_rows_added = 0
        for igap in wgap: 
##            igap += total_rows_added
##            print '***', igap, diff[igap], times[igap:igap+2]
##            print (data_this_station['time'])[igap:igap+2]

            # How many rows do we need to add? (i.e. how big was the gap?)
            ntimes_to_add = diff[igap] - 1

            # Figure out the times we must add, and convert back to seconds
            times_to_add = numpy.arange(ntimes_to_add) + times[igap] + 1
            times_to_add *= SECONDS_PER_MINUTE

            # Most of the original timing data is a few seconds after the 
            # start of the minute.  Let's make these new times 1 second
            # after.
            times_to_add += 1

            # Add rows to the data table.  First define the new values for
            # each column.
            ids = numpy.full(ntimes_to_add, station_id)
            ba = numpy.full(ntimes_to_add, 
                            -1)
##                            (data_this_station['bikes_available'])[igap])
            da = numpy.full(ntimes_to_add, 
                            (data_this_station['docks_available'])[igap])
            mytime = times_to_add.astype('datetime64[s]')

            my_t_norm = (data_this_station['t_norm'])[igap] + \
                (times_to_add - 
                 data_this_station['time'][igap].astype(int)) / \
                 SECONDS_PER_MINUTE

            # Correct t_norm to ensure it's less than 1 week
            wbad = my_t_norm > MINUTES_PER_WEEK
            my_t_norm[wbad] = my_t_norm[wbad] - MINUTES_PER_WEEK

            # Actually add rows to table.
            keys = ['station_id', 'bikes_available','docks_available',
                    'time', 't_norm']
            values = [ids, ba, da, mytime, my_t_norm]
#            insert_index = -1
            insert_index = (numpy.where(w_id)[0])[0] + \
                igap + 1 + \
                total_rows_added

            add_rows_to_table(self.rebalancing_data, values,
                              insert_index = insert_index, keys = keys)

            total_rows_added += ntimes_to_add                
        pass

    ####################################
    def filter(self, station_id):
        """Fourier filter a station's time series data"""
        pass

    ####################################
    def plot_filter(self, station_id, max_freq=10000):
        """Fourier filter a station's time series data"""
        filtered = self.one_filter(station_id, max_freq, doplot=1)


        # Plot the filtered time series
        xlim = [0,10000]
        xlabel = 'time [minutes]'
        ylabel = '# bikes'
        pyplot.subplot(311)
        pyplot.plot(filtered)
        pyplot.xlim(xlim)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel+' FFT')
        
        # Plot the original time series
        pyplot.subplot(312)
        pyplot.plot(self.temp_timeseries)
        pyplot.xlim(xlim)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)

        # Plot the difference
        pyplot.subplot(313)
        pyplot.plot(self.temp_timeseries - filtered)
        pyplot.xlim(xlim)
        pyplot.xlabel(xlabel)
        pyplot.ylabel('Difference')
        pyplot.show()


    ####################################
    def one_fourier(self, station_id, power=True, shift=True):
        """Get Fourier transform for one time series"""
        quantity = 'bikes_available'

        # Find bikes matching this station ID
        w_id = self.rebalancing_data['station_id'] == station_id

        # Extract the appropriate time series and take its Fourier transform
        timeseries = self.rebalancing_data[w_id][quantity]

        # Save (temporarily) for later use (e.g.plotting)
        self.temp_timeseries = timeseries

        # Take the FFT
        myfft = FFT.fft(timeseries)
        if shift:
            myfft = FFT.fftshift(myfft)

        self.temp_ftt = myfft

        # Should we return the power spectrum or just the FFT?
        if power:
            power = abs(myfft)**2.0
            self.temp_power_spec = power
            myfft = power

        return myfft
            

    ####################################
    def one_filter(self, station_id, max_freq=500, doplot=0):
        """Fourier filter one time series"""
        quantity = 'bikes_available'
        
        myfft = self.one_fourier(station_id, power=False, shift=True)
        npoints = len(myfft)
        freq_x = numpy.arange(npoints) - npoints / 2
        
#         pyplot.plot(myfft)
#         pyplot.show()

        # zero out regions of the fourier series at large frequencies
        w_zero = abs(freq_x) > max_freq
        filt_fft = myfft.copy()
        filt_fft[w_zero] = 0.0

##        print "About to filter"

        # Inverse fourier transform to get the original time series, but
        # filtered. (Also make sure it's de-shifted before taking inverse.)
        filtered_timeseries = FFT.ifft(FFT.ifftshift(filt_fft))

##        print "Done filter"
        if doplot > 0:
            xdat = freq_x
            xlim = 10.*numpy.array([-1*max_freq, max_freq])
            pyplot.figure(1)
            pyplot.subplot(211)
            pyplot.plot(xdat, myfft)
            pyplot.xlim(xlim)
            pyplot.subplot(212)
            pyplot.plot(xdat, filt_fft)
            pyplot.xlim(xlim)
            pyplot.show()
           
        print "Done plotttt"
        return filtered_timeseries

    ####################################
    def one_autocorr(self, station_id):
        """Autocorrelate one time series (with itself)"""
        quantity = 'bikes_available'

        # Find bikes matching this station ID
        w_id = self.rebalancing_data['station_id'] == station_id

        # Extract the appropriate time series and take its Fourier transform
        timeseries = self.rebalancing_data[w_id][quantity]

        # Save (temporarily) for later use (e.g.plotting)
        self.temp_timeseries = timeseries

##        corr = numpy.correlate(timeseries, timeseries, mode='full')

        corr = signal.fftconvolve(timeseries, timeseries[::-1], mode='full')
        autocorr = corr[len(corr)/2:]
        xdat = numpy.arange(len(autocorr)) / 60.0
        
        pyplot.plot(xdat, autocorr)
        pyplot.show()
        return autocorr

    ####################################
    def neighbor_dist_ts(self, station_id):
        """Find stations with similar time series using Euclidean distance 
        measure.

        Return a tuple of (neighbor_ids, distances) where distances is 
        a sorted array of Euclidean distances between one timeseries and
        all the other timeseries, and neighbor_ids gives the corresponding
        station IDs.  The first station in the list will be itself (which
        has a zero distance).
        """

        # Get normalized, filtered time series for station of interest
        ts1 = self.get_clean_ts(station_id)
        
        all_IDs = numpy.unique(self.rebalancing_data['station_id'])

        # Create array to store the sorted distances (and another to store
        # the station IDs ranked from nearest to farthest)
        LARGE_VALUE = 1e10
        dis = numpy.zeros(len(all_IDs)) + LARGE_VALUE
        neighbor_ids = numpy.zeros(len(all_IDs))

        # Loop over all other stations
        print "Looping over neighbors of station ", station_id
        for i, myID in enumerate(all_IDs):
#             print i,
#             sys.stdout.flush()

            # We don't compare the station of interest to itself
            if myID == station_id:
                dis[i] = 0.0
                neighbor_ids[i] = myID
                continue

            # get the filtered time series for this station
            ts2 = self.get_clean_ts(myID)
            
            # Determine the 'distance' (using an arbitrary metric)
            # between this station and the station of interest
###            dd = compare_time_series(filtered_main, filtered_neighbor)
            euclid_dis = ts1.euclidean_distance(ts2)

            dis[i] = euclid_dis
            neighbor_ids[i] = myID

            pass

        # Now, sort the results
        # (NB: for faster results when seeking only the N-nearest neighbors,
        #  it would be better to 'sort-as-you-go' within the loop, and 
        #  avoid sorting neighbors that are not within the current N nearest).
        ss = numpy.argsort(dis)
        dis = dis[ss]
        neighbor_ids = neighbor_ids[ss]

        return neighbor_ids, dis

    ####################################
    def neighbor_dist_ts_all(self):
        """Calculate Euclidean distance measure for each pair of stations"""
        neighbor_list = []
        print "Looping over all stations, finding their neighbors..."
        for station_id in numpy.unique(self.station_data['station_id']):
            print station_id,
            sys.stdout.flush()

            # Find the distance to each other station's time series
            neighbor_ids, dist = self.neighbor_dist_ts(station_id)

            # Save for later
            neighbor_list.append(Neighbor(neighbor_ids, dist))

        self.neighbor_list = neighbor_list

    ####################################
    def neighbor_days(self, station_id, time):
        """Given a station and a day/time, find days with similar time series
        among both a) the same station's other days and b) neighbor stations'
        days.

        Something must be done already -- clean_ts
        
        Result should be an array of (stat_id, index) pairs, where
        index gives the index to the rebalancing_data sub-table (for the
        station indicated by stat_id) that is the *start* of a 24-hour 
        period that is a close neighbor to the day-of-interest.
        """

        # Get the time series for this day and station
        main_day = DayTS(time, station_id, self)
        
        # Find index of each minute that is the first minute of the day
        w_daystart = self.find_first_minute()
        daystart_ind_arr = numpy.where(w_daystart)[0]

        # output arrays.  'start_inds' refers to the index to common_timeline
        # that gives the first minute of the appropriate day.
        LARGE_VALUE = 1e20
        neighbor_day_station_ids = numpy.array()
        neighbor_day_start_inds = numpy.array()
        neighbor_day_dis = numpy.zeros(n_neighbors) + LARGE_VALUE

        # Loop over this station's neighbor stations.
        wid = self.station_data['station_id'] == station_id
        for neighbor_id in neighbor_list[wid].IDs:
            # Loop over the neighbor station's days
            for daystart_ind in daystart_ind_arr:
                time = self.common_timeline[daystart_ind]
                day = DayTS(time, neighbor_id, self)

                # Calculate distance between day-of-interest and this
                # neighbor station's day.
                dist = main_day.metric_distance(day)
                
                # If this is a close neighbor, insert it into the array
                # of close neighbors in sorted order.
                if dist < neighbor_day_dis[n_neighbors]: 
                    ind = numpy.searchsorted(neighbor_day_dis, dist)
                    neighbor_day_dis=numpy.insert(neighbor_day_dis, ind, dist)
                    neighbor_day_station_ids = numpy.insert(
                        neighbor_day_station_ids, ind, neighbor_id)
                    neighbor_day_start_inds = numpy.insert(
                        neighbor_dat_start_inds, ind, daystart_ind)
                    
                
        # Clean up by getting rid of extra days in output arrays
        neighbor_day_dis = neighbor_day_dis[0:n_neighbors]
        neighbor_day_station_ids = neighbor_day_station_ids[0:n_neighbors]
        neighbor_day_start_inds = neighbor_day_start_inds[0:n_neighbors]
        return nieghbor_day_station_ids, neighbor_day_start_inds,\
            neighbor_day_dis

    ####################################
    def get_clean_ts(self, station_id, quantity='bikes_available'):
        """Get a normalized, filtered time series for a given station"""

        w_id = self.rebalancing_data['station_id'] == station_id

        # Normal operation is to filter then normalize the time series
        if self.clean_ts_arr is None:
            # Get the basic time series
            self.temp_timeseries = self.rebalancing_data[w_id][quantity]
            ts1 = TimeSeries(self.temp_timeseries)

            # Low-pass filter the time series
            ts1.filter()
            
            # Normalize the time series.  The max value should be the
            # total number of bike docks (aka stands) at the station, 
            # which we get from the station_data structure.
            w_id_stat = self.station_data['station_id'] == station_id
            n_stands = self.station_data['dockcount'][w_id_stat]
            ts1.normalize(maxval= n_stands, minval = 0)
        
        else:
            # If we already have a normalized time series for each station
            # then we just use that.
            ts1 = TimeSeries(self.clean_ts_arr[w_id])

        return ts1

    ####################################
    def clean_ts_all(self):
        """Create an array of normalized, filtered time series including
        all bike stations
        """
        unique_ids = numpy.unique(self.rebalancing_data['station_id'])
##        n_station = len(unique_ids)
        temp_arr = numpy.zeros(len(self.rebalancing_data))
        
        # Loop over stations
        print "Cleaning time series"
        for station_id in unique_ids:
            print station_id,
            sys.stdout.flush()
            ts = self.get_clean_ts(station_id)
            w_id = self.rebalancing_data['station_id'] == station_id
            temp_arr[w_id] = ts.data

        # Assign the resulting array to the class instance variable
        self.clean_ts_arr = temp_arr

    ####################################
    def find_first_minute(self):
        """Using the common timeline, find the index of the first minute
        of each day"""
        times = self.common_timeline
        
        # Set up an empty array for the return values.
        w_daystart = numpy.zeros(len(times), dtype='bool')
        for i, time in enumerate(times):
            dt = datetime.datetime.fromtimestamp(time.astype('int'),timezone)
            if dt.minute == 0 and dt.hour == 0:
                w_daystart[i] = True

        return w_daystart

    ####################################
    def build_station_list(self, load_filename=None):
        """Build a list of stations
        
        Create a list of Station objects corresponding to all the stations
        in this bike system.  Each Station object 

        NB: This actually stores the stations in a StationGroup object,
        the main advantage of which is the get_station() method, which
        allows you to get a given station based on its station_id (rather
        than trying to mess with index values in a table)
        """
        # If loadfile is provided, then we just read the station data from
        # a file rather than creating new stations.
        if load_filename is not None:
            self.load_stations(load_filename)
            return
        
        ##### Below we construct new stations (rather than reading from file)
        station_list = []
        for station_id in numpy.unique(self.station_data['station_id']):
            station = Station(station_id, self.rebalancing_data,
                              self.station_data, timeline=self.common_timeline,
                              weather=self.weather)
            station_list.append(station)

        self.all_stations = StationGroup(station_list)


    ####################################
    def friends_of_friends(self, linking_length = 0.00024):
        """Create clusters of stations via Friends-of-Friends algorithm
        
        """
        all_stations = self.all_stations.station_list

        # Set all cluster IDs to 0
        for station in all_stations:
            station.set_cluster_id(0)

        cluster_ids = numpy.zeros(len(all_stations))
        current_cluster_id = 1
        
        # loop over all stations, finding "friends" of each
        for i, station in enumerate(all_stations):
            cluster_ids[i] = station.cluster_id

            # If we already put this station in a cluster, then
            # skip it.
            if cluster_ids[i] == 0:
                # Find friends of this station.  This will be called
                # recursively, finding all friends-of-friends.  All
                # stations will have cluster_id set to current_cluster_id
                # 
                station.find_friends(linking_length, current_cluster_id)

                # Increment cluster ID number
                current_cluster_id += 1
                
        
        print "Found clusters:", current_cluster_id-1, len(all_stations)


    ####################################
    def kmeans(self, K_clusters=10):
        """Find clusters of stations using K-means clustering 

        K_clusters: the number of clusters to search for
        """
        #### Create a big matrix of timeseries data to feed into the
        # K-means algorithm.
        # 'ndim' is the number of dimension, i.e. the number of time points
        # in the time series.
        n_stations = len(self.all_stations.station_list)
        ndim = len(self.common_timeline.times)
        dat = numpy.zeros((n_stations, ndim))

        # Loop over all stations, adding each one's timeseries data
        # to the dat variable.
        for i, station in enumerate(self.all_stations.station_list):
            dat[i,:] = station.ts_norm

            
        # Clean up the data by getting rid of "bad" values 
        for ii in range(len(dat[:,0])):
            wbad = dat[ii,:] <= -1
            replacement_value = 0.5  # Should use some mean here?
            dat[ii,wbad] = replacement_value
            pass

        #### Now run K-means algorithm on the data matrix
        MaxIter = 25
        centroids, idx = kmeans.runKmeans(dat, K_clusters, MaxIter)

        return centroids, idx


    ####################################
    def regress_all_stations(self):
        """Run regression on each station
        """
        stationlist = self.all_stations.station_list
        for station in stationlist:

#             if station.ID > 2:
#                 break

            print "================================================"
            # Run first set of regressions on 1000 examples, finding the 
            # best C and gamma for that many examples
            station.regress(subsample_size = 1000,
                            selection_criterion = 'training')

            # Now try with 3000 examples, but a smaller range of 
            # C and gamma values
            C_best = station.svr_fit.get_params()['C']
            gamma_best = station.svr_fit.get_params()['gamma']
            C_grid = numpy.array([1.0, 10.0]) * C_best
            gamma_grid = numpy.array([1.0, 10.0, 100.0]) * gamma_best
            station.regress(C_grid, gamma_grid, subsample_size=3000,
                            selection_criterion = 'training')
            
            # Now do the final regression with 20000 examples
            C_best = station.svr_fit.get_params()['C']
            gamma_best = station.svr_fit.get_params()['gamma']
            C_grid = [C_best]
            gamma_grid = numpy.array([1.0,10.0]) * gamma_best
            station.regress(C_grid, gamma_grid, subsample_size=20000,
                            selection_criterion = 'crossval')


####################################
def quick_load_sfbd():
    """Quickly load San Francisco bike data from save files
    """
    sfbd = SFBikeData(unpickle=1, timeline_filename='common_timeline.pkl',
                      stationlist_filename = 'stationlist.pkl')
    return sfbd


####################################
####################################
class SFBikeData(BikeData):
    """Class for San Fran bike sharing data"""

    ####################################
    def setup_vars(self):
        self.dir = '/Users/jgabor/Documents/Junk/VELIB/' + \
            'SF_bikeshare/201408_babs_open_data/'
        

    ####################################
    def read_rebalance(self, unpickle=0, save=0):
        """Read SF bike rebalancing data"""

        # Read original data directly from the ascii file, then do some
        # manipulations to new data fields.
        if unpickle == 0:
            filename = self.dir + '201408_rebalancing_data.csv'

            print "reading ASCII data from file "+filename
            dat = sf_bike.read_rebalance(filename, unpickle=unpickle)
            print "done reading data."

            print "Adjusting time data"
            dd = sf_bike.adjust_times(dat)
            print "done adjusting"
            
            print "Checking for gaps in time series"
            self.rebalancing_data = dd
            self.check_gaps_all()
            print "done gap checking"


        # Just read the previously pickled data
        elif unpickle > 0:
            pick_filename = self.dir + "sfdat.pkl"
            print "Reading data from file "+pick_filename
            ff = open(pick_filename, 'rb')
            dd = pickle.load(ff)
            ff.close()
            print "...done"

        return dd

    ####################################
    def read_station(self):
        """Read SF bike station data"""
        filename = self.dir + "201408_station_data.csv"
        print "Reading station data from ascii file " + filename
        dat = sf_bike.read_station(filename)
        return dat
        pass

    ####################################
    def read_clean_ts(self, unpickle = 0):
        """This should be redefined by specific subclasses"""
        filename = self.dir + 'clean_ts.pkl'

        
        if unpickle > 0:
            if os.path.isfile(filename) is False:
                print "No clean_ts file to read..."
                return None
            print "Reading clean_ts data from pickle file "+ filename
            file = open(filename, 'rb')
            clean_ts_dat = pickle.load(file)
            file.close()
            print "...done"
        else: 
            clean_ts_dat = None

        return clean_ts_dat


####################################
####################################
class TimeSeries:
    """Class for time series analysis"""
    
    ####################################
    def __init__(self, timeseries_dat, save_orig=False):
        self.data = timeseries_dat
        if save_orig:
            self.orig_data = timeseries_dat.copy()
        else:
            self.orig_data = None
        pass

    ####################################
    def normalize(self, maxval=None, minval=None,
                  normtype = 1, badval = -1, newbadval = -1):
        """Normalize a time series so that its values fall between 
        0.0 and 1.0"""
        timeseries = self.data
        wbad = timeseries == badval

        # In this case, normalize so that the time series values fluctuate
        # between zero and 1.0.
        if normtype == 1:
            if maxval is None:
                maxval = numpy.max(timeseries)
            if minval is None:
                minval = numpy.min(timeseries)
            norm_factor = (maxval - minval)

        # This is the more standard method where we subtract out the mean
        # and divide by the standard deviation.
        if normtype == 2:
            if maxval is None:
                maxval = numpy.std(timeseries)
            if minval is None:
                minval = numpy.mean(timeseries)
            norm_factor = maxval  ### == standard deviation

        # Do the normalization.
        norm_timeseries = timeseries.astype('float') - minval
        norm_timeseries /= float(norm_factor)    ### (maxval-minval)
                
        norm_timeseries[wbad] = newbadval

        # redundant?  (No I don't think so.)
        self.data = norm_timeseries

        return norm_timeseries


    ####################################
    def euclidean_distance(self, timeseries2, badval= -1):
        """Compare another time series with this one.
        Both series should probably be normalized first.
        """
        NPIX_MIN = 100

        # Time series assumed to be the same length
        ts1 = self.data
        ts2 = timeseries2.data

        #####################
        # If one timeseries is shorter than the other, fill in the shorter
        # one with bad pixels
        len1 = len(ts1)
        len2 = len(ts2)
        if len1 > len2:
            temp = ts2
            ts2 = numpy.full_like(ts1, badval)
            ts2[0:len2] = temp
        if len2 > len1:
            temp = ts1
            ts1 = numpy.full_like(ts2, badval)
            ts1[0:len1] = temp

        # Bad pixels should not contribute to the Euclidean distance
        wbad1 = ts1 == badval
        wbad2 = ts2 == badval

        # Find the squared difference array, and correct for bad pixels
        ED_arr = (ts2 - ts1)**2.0
        ED_arr[wbad1] = 0
        ED_arr[wbad2] = 0

##        print 'PPPPPPPPPP', numpy.min(ED_arr), numpy.max(ED_arr)
#         pyplot.plot(ED_arr)
#         pyplot.show()

        # Find Euclidean distance
        ED = numpy.sqrt(numpy.sum(ED_arr) )

        # Normalize the Euclidean distance by the number of good pixels
        n_good_pix = len(self.data) - \
            numpy.count_nonzero(numpy.logical_or(wbad1,wbad2))
        
        LARGE_VALUE = 1e20
        if(n_good_pix) < NPIX_MIN:
            ED = LARGE_VALUE
        else:
            ED /= n_good_pix
        
        return ED


    ####################################
    def filter(self, max_freq=500, doplot=0, badval = -1):
        """Fourier filter one time series"""
        wbad = self.data == badval

        myfft = self.fourier(power=False, shift=True)
        npoints = len(myfft)
        freq_x = numpy.arange(npoints) - npoints / 2
        
        # zero out regions of the fourier series at large frequencies
        w_zero = abs(freq_x) > max_freq
        filt_fft = myfft.copy()
        filt_fft[w_zero] = 0.0

##        print "About to filter"

        # Inverse fourier transform to get the original time series, but
        # filtered. (Also make sure it's de-shifted before taking inverse.)
        filtered_timeseries = FFT.ifft(FFT.ifftshift(filt_fft))

        # Take the real part only.  The imaginary parts should already
        # be small.
##        filtered_timeseries = numpy.real_if_close(filtered_timeseries)
        filtered_timeseries = numpy.real(filtered_timeseries)

        if numpy.max(numpy.imag(filtered_timeseries)) > 1e-20:
            raise Exception("Bad filtration: imaginary values still " + \
                                "present {0}".
                            format(numpy.max(numpy.imag(
                            filtered_timeseries)) ) )

        # Make sure "bad" values remain the same.
        filtered_timeseries[wbad] = badval

##        print "Done filter"

        if doplot > 0:
            xdat = freq_x
            xlim = 10.*numpy.array([-1*max_freq, max_freq])
            pyplot.figure(1)
            pyplot.subplot(211)
            pyplot.plot(xdat, myfft)
            pyplot.xlim(xlim)
            pyplot.subplot(212)
            pyplot.plot(xdat, filt_fft)
            pyplot.xlim(xlim)
            pyplot.show()
            print "Done plotttt"

        self.data = filtered_timeseries
        return filtered_timeseries

    ####################################
    def fourier(self, power=True, shift=True):
        """Get Fourier transform for time series"""

        # Extract the appropriate time series and take its Fourier transform
        timeseries = self.data

        # Take the FFT
        myfft = FFT.fft(timeseries)
        if shift:
            myfft = FFT.fftshift(myfft)

        # Should we return the power spectrum or just the FFT?
        if power:
            power = abs(myfft)**2.0
            self.temp_power_spec = power
            myfft = power

        return myfft

    ####################################
    def fold(self, period, start_index=0,
             fill_incomplete=True, cut_incomplete=False,
             badval = -1):
        """Periodically fold a time series
        
        NB: This will not account correctly for e.g. daylight
        savings time!

        XXXX Returns an index array.
        Now returns an array of dimension (nperiods,npoints) array,
        with nperiods different segments each an npoints time series
        array.
        """
        ndat = len(self.data)

#         # The +1 is to ensure we have enough space 
#         # for all the indexes.
#         nperiods = ndat / period + 1

#         # Generate an nperiods by period array
#         ygrid = numpy.mgrid[0:nperiods, 0:period][1]

#         # Reshape the array to get a periodic array of indexes
#         # with the specified period
#         index = numpy.reshape(ygrid, nperiods*period)
        
#         print "SHAPE", numpy.shape(ygrid), numpy.shape(index)
# ##        raise Exception("junk")
        
#         # Reduce to the correct number elements
#         index = index[0:ndat]

#         print "SHAPE2", numpy.shape(index)

#         # Shift the index values as desired
#         index = numpy.roll(index, -1*start_index)

###        return index

        # In this case, we typically shift the data so that the first
        # element is at the beginning of 1 period, then we remove
        # the last period, which is only a partial period.
        if cut_incomplete is True:
            fill_incomplete = False

            # Shift the data so daystart is at beginning of array
            timeser = numpy.roll(self.data, -1*start_index)

            # cut off extra data at the end of time series 
            # (the incomplete period at the end)
            n_full_periods = ndat / period
            timeser = timeser[0: n_full_periods*period]
        
            outarr = numpy.reshape(timeser, (n_full_periods, period))
            pass

        # In this case we generally do no data shifting, and we fill
        # in values in the first and last period with NANs or negative
        # values
        if fill_incomplete is True:
            # We need the +1 to ensure there's enough space for partial
            # periods.
            n_periods = ndat / period + 2
            
            # Create an empty array to hold the data
            mytype = self.data.dtype
            if badval is None:
                if mytype == numpy.dtype('int'):
                    badval = -1
                else:
                    badval = -1.0
##                    badval = numpy.nan

            arr = numpy.full(n_periods * period, badval, dtype=mytype)
            
            # Insert the real data, starting at start_index
            arr[start_index:start_index+ndat] = self.data

            # Now fold
            outarr = numpy.reshape(arr, (n_periods, period))
            pass

        return outarr
        pass

    ####################################
    def seasonal_correct(self, period, start_index=0, plot=0):
        """Do a seasonal correction of periodic signal

        period: in minutes.  e.g. 1 day = 1440 minutes
        """
        # fold the time series by the desired period
        foldarr = self.fold(period, start_index)

        # Get the median and std. deviation timeseries for this 
        # period
        ## med = numpy.mean(foldarr,0)
        med = numpy.median(foldarr,0)
        std = numpy.std(foldarr,0)

        # Subtract the median and divide out the std. deviation
        new_arr = foldarr.copy()
        for i,dayarr in enumerate(foldarr):
            new_arr[i] = (dayarr - med) / std
            wbad = dayarr < 0
            new_arr[i][wbad] = -1

        arr = new_arr.reshape(new_arr.size)

        if plot > 0:
            # get the autocorrelation function and plot it
            corr = signal.fftconvolve(arr, arr[::-1], mode='full')
            autocorr = corr[len(corr)/2:]
            xdat = numpy.arange(len(autocorr)) / 60.0
        
            pyplot.plot(xdat, autocorr)
            pyplot.show()

        return arr

####################################
####################################
class Neighbor:
    """Class to store information about an object's neighbors"""
    def __init__(self, IDs, distances):
        """
        IDs: a (sorted) array of neighbor IDs
        distances: a (sorted) array of neighbor distances
        """
        self.IDs = IDs
        self.dist = distances


####################################
####################################
class DayTS_ORIG:
    """Class to handle operations on a timeseries for 1 day"""
    def __init__(self, time, station_id = None, bikedat = None):
        """ 
        time is a datetime64 object
        station_id must be specified if a Station object is not given
        bikedat is a BikeData object
        """
        self.station = station_id
        wid = bikedat.rebalancing_data['station_id'] == station_id
        station_times = bikedat.rebalancing_data['time'][wid]

        self.ind_this_station = numpy.where(wid)[0][0]
        
        timezone = sf_bike.USTimeZone()

        # Given 'time', figure out the first minute of the calendar
        # day in which this time falls
        dt = datetime.datetime.fromtimestamp(time.astype('int'), timezone)
        dt_daystart = dt.replace(hour=0, minute=0)
        
        # Convert this back to a datetime64 object for easy comparison
        dt_start = numpy.datetime64(dt_daystart.isoformat(),'s')

        # Find the end of the day, 24 hours later
        dt_end = dt_start + numpy.timedelta64(24,'h') - \
            numpy.timedelta64(1,'m')
        
        # Find the best time value in the table that is closest to this
        # prescribed minute (the minute at the start of the day)
        self.ind_start = numpy.argmin(numpy.abs(station_times - dt_start)) + \
            self.ind_this_station
        self.ind_end = numpy.argmin(numpy.abs(station_times - dt_end)) + \
            self.ind_this_station
        
#         print dt_start, dt_end
#         print datetime.datetime.fromtimestamp(dt_start.astype('int'), timezone)
#         print ind_start, ind_end

        # Data from the original table that corresponds to this station
        # and time.
        # NB: this is a pointer to the original table -- if you modify it,
        #  it will modify the original table
        self.table = bikedat.rebalancing_data[self.ind_start:self.ind_end]
        self.table_all = bikedat.rebalancing_data
        self.clean_ts = bikedat.clean_ts_arr[self.ind_start:self.ind_end]

    ####################################
    def metric_distance(self, otherday):
        """Find the distance between this day and another, where distance
        is defined using a given metric, e.g. the Euclidean distance between
        their time series data, the physical distance on the surface of the 
        Earth, or some weighted combination of the two.

        otherday: a different DayTS object
        """
        # Create TimeSeries object for each day
        ts1 = TimeSeries(self.clean_ts_arr)
        ts2 = TimeSeries(otherday.clean_ts_arr)

        # Measure Euclidean distance between these days
        euclid_dist = ts1.euclidean_distance(ts2)

        dist = euclid_dist

        return dist
    


####################################
####################################
class StationGroup:
    """Class to hold many Station objects"""

    ####################################
    def __init__(self, station_list = []):
        self.station_list = station_list

        # Find each station's ID and put it in a list
        self.station_ids = []
        for station in station_list:
            self.station_ids.append(station.ID)

        pass

    ####################################
    def add_station(self, station):
        # Add station to the list
        self.station_list.append(station)

        # Add station's ID
        self.station_ids.append(station.ID)

    ####################################
    def get_station(self, station_id):
        """Return one of the stations from the station list"""
        if station_id in self.station_ids:
            ind = self.station_ids.index(station_id)
            ss = self.station_list[ind]
            pass
        else:
            print "Station not found in StationGroup list!"
            ss = None

        return ss
        


####################################
####################################
class Station:
    """Class to hold all data about a station"""

    ####################################
    def __init__(self, station_id, rebalancing_data, station_data,
                 timeline = None, loadfile = None, weather=None):
        """
        rebalancing_data: a data table for all stations
        station_data: a data table for all stations
        timeline: should be the common timeline for all stations
        
        loadfile: An already-opened pickle file with data for many
          stations.  If provided, then we will use this file to load
          important data rather than creating new data structures.  In this
          case station_id, rebalancing_data, and station_data values
          do not matter and are not used, but timeline must
          be provided.
        """
        
        # If loadfile is set, then we want to load the station data
        # from a file.
        if loadfile is not None:
            # Attach this station to a timeline.
            if timeline is None:
                raise Exception("No timeline provided, but loadfile provided")
            else:
                self.timeline = timeline

            self.load_data(loadfile)

        else:
            # We're creating a new station object (as opposed to loading
            # data from a file)
            self.ID = station_id
        
            # Find appropriate station data from the bikedat structure
            # NB: This makes a COPY of the data (it's not just a pointer
            # to the original data)
            w_id = rebalancing_data['station_id'] == station_id
            self.table = rebalancing_data[
                'bikes_available','docks_available'][w_id]

            # Get basic data such as location, number of docks. 
            # Note that this makes a copy of the station_data, it is not
            # just a view of it.
            wID = station_data['station_id'] == station_id
            self.basic_data = station_data[wID]
        
            # Attach this station to a timeline.
            if timeline is None:
                self.timeline = Timeline(rebalancing_data['time'][w_id])
            else:
                self.timeline = timeline

            self.extra_features = None
            self.feature_norm_params = None
            self.svr_fit = None
            self.minerr = None
            self.ts_norm = None
            self.ts_clean = None

        #######
        # All the remaining attributes are set whether we loaded data from
        # a file or whether we created new data structures.

        self.weather = weather

        # Set data subsample indexes to None so we know they haven't
        # been determined yet
        self.training_ind = None
        self.cross_validation_ind = None
        self.test_ind = None

        self.daylist = None
        self.neighbor_dist = None
        self.neighbor_list = None
        self.cluster_id = None

        self.bootstrap_list = None
        self.dat_list = None
        self.bootstr = None


    ####################################
    def quicksave(self, filename):
        file = open(filename,'wb')
        self.save_data(file)
        file.close()

    ####################################
    def save_data(self, file_obj):
        """Save underlying data for this Station

        file_obj -- should be an open file object, not a filename! This
          way we can save many stations in one file.
        """
        # Open file for output
        ##file = open(filename, 'wb')
        # Rename for convenience.
        file = file_obj

        # Save the station ID
        pickle.dump(self.ID, file, -1)

        # Save the main data table
        pickle.dump(self.table, file, -1)
        
        # Save the station data
        pickle.dump(self.basic_data, file, -1)

        # Save regression fit data
        pickle.dump(self.svr_fit, file, -1)
        pickle.dump(self.minerr, file, -1)
        
        # Save indices to the different regression data sets
        pickle.dump(self.training_ind, file, -1)
        pickle.dump(self.cross_validation_ind, file, -1)
        pickle.dump(self.test_ind, file, -1)
        pickle.dump(self.feature_norm_params, file, -1)
        pickle.dump(self.extra_features, file, -1)

        return 0
        

    ####################################
    def load_data(self, file_obj):
        """Load saved station data 
        """
        file = file_obj

        # Load station ID
        self.ID = pickle.load(file)
        if type(self.ID) != numpy.int8:
            raise Exception("loaded bad station ID: type is "+
                            str(type(self.ID)) )
        
        self.table = pickle.load(file)
        self.basic_data = pickle.load(file)
        self.svr_fit = pickle.load(file)
        self.minerr = pickle.load(file)
        self.training_ind = pickle.load(file)
        self.cross_validation_ind = pickle.load(file)
        self.test_ind = pickle.load(file)
        self.feature_norm_params = pickle.load(file)
        self.extra_features = pickle.load(file)

        # Now we're done loading data from the file, but we want
        # to set some other variables derived from those data.

        if 'ts_norm' in self.table.colnames:
            self.ts_norm = self.table['ts_norm']
        else:
            self.ts_norm = None

        if 'ts_clean' in self.table.colnames:
            self.ts_clean = self.table['ts_clean']
        else:
            self.ts_clean = None



    ####################################
    def create_days(self):
        """Create a bunch of DayTS structures for this station"""
        daylist = []
        
        # Loop over day start times in the timeline
        for ii, ind_daystart in enumerate(self.timeline.ind_daystart):
            day = DayTS(ind_daystart, self, ID = ii)
            daylist.append(day)

        self.daylist = daylist
        return daylist

    ####################################
    def get_days(self, day_index, startdate=None, enddate=None):
        """Retrieve certain days of the week from 'daylist'

        day_index: Monday == 0, Sunday == 6
        startdate: A datetime object. Days occuring before the specified
           date will be ignored.
        """
        if startdate is None:
            startdate = datetime.date(1900, 1,1)

        if enddate is None:
            enddate = datetime.date(2100,1,1)

        result = []
        for day in self.daylist:
            # Skip days outside the defined dates.
            if day.date < startdate or day.date > enddate:
                continue

            # Match to the day of the week.
            if day.day_of_week_index == day_index:
                result.append(day)

        return result
    

    ####################################
    def normalize_ts(self):
        """Normalize this station's bikes_available time series
        """
        # Create a TimeSeries object
        ts = TimeSeries(self.table['bikes_available'])
        
        maxval = self.basic_data['dockcount']
        minval = 0
        ts_norm = ts.normalize(maxval = maxval, minval = minval)

        # Add the normalized timeseries to the station data
        self.table['ts_norm'] = ts_norm
        self.ts_norm = self.table['ts_norm']

#        self.ts_norm = ts_norm
        return ts_norm

    ####################################
    def clean_ts(self):
        """Normalize and filter this station's time series"""
        # Normal operation is to filter then normalize the time series

        # Create TimeSeries object
        ts = TimeSeries(self.table['bikes_available'])

        # Low-pass filter
        ts.filter()

        # Normalize
        maxval = self.basic_data['dockcount']
        minval = 0
        ts.normalize(maxval = maxval, minval=minval)

        # Add the cleaned timeseries as ts_clean to the data table
        self.table['ts_clean'] = ts.data

        self.ts_clean = self.table['ts_clean']
        return ts.data

    ####################################
    def find_neighbors(self, other_station_list):
        """Make a list of nearest neighbor stations"""
        dist_all = []
        for other_station in other_station_list:
            dist = self.compare_stations(other_station)
            dist_all.append(dist)
            
        # Sort by distance
        dist_arr = numpy.array(dist_all)
##        ss = numpy.sort(dist_arr)
        ss = numpy.argsort(dist_arr)
        neighbor_list = []
        for ii in range(len(dist_all)):
            neighbor_list.append( other_station_list[ss[ii]])

        self.neighbor_dist = dist_arr[ss]
        self.neighbor_list = neighbor_list
        
    ####################################
    def compare_stations(self, other_station):
        """Compare stations' timeseries"""
#         ts1 = TimeSeries(self.ts_clean)
#         ts2 = TimeSeries(other_station.ts_clean)
        ts1 = TimeSeries(self.ts_norm)
        ts2 = TimeSeries(other_station.ts_norm)

        euclid_dist = ts1.euclidean_distance(ts2)

        dist = euclid_dist
        return dist

    ####################################
    def plot_ts(self):
        """Plot various versions of the time series"""
##        xx = self.timeline.times
        xx = self.timeline.datetimes

        # Set up stuff for x-axis tick labels
        loc_months = mdates.MonthLocator(tz=self.timeline.timezone)
        monthFmt = mdates.DateFormatter("%b")

        pp = None
        if 'ts_norm' in self.table.colnames:
            yy = self.table['ts_norm']
            pp = pyplot.plot(xx, yy)

        if 'ts_clean' in self.table.colnames:
            yy = self.table['ts_clean']
            pp = pyplot.plot(xx,yy)

        if pp is None:
            print "Neither ts_norm nor ts_clean is set: can't plot!!!"
        else:
            ax = pyplot.gca()
            ax.xaxis.set_major_locator(loc_months)
            ax.xaxis.set_major_formatter(monthFmt)

            pyplot.ylabel("Fraction of bikes available")
            pyplot.show()


    ####################################
    def bootstrap_day_of_week(self, startdate=None, enddate=None):
        """Do a bootstrap sampling by the day of week

        """

        ndays_per_week = 7
        bootstrap_list = []
        dat_list = []

        # Loop over days of the week
        for day_ind in range(ndays_per_week):
            # Find all points on the timeline that represent the start
            # of this day of the week.
            daylist = self.get_days(day_ind, startdate, enddate)

            ndays = len(daylist)
            ntimes_per_day = len(daylist[0].ts)

            # Create 2D data array where each row is a single day.
            # 
            dat = numpy.zeros((ndays, ntimes_per_day))

            for iday, day_obj in enumerate(daylist):
                # Get the time series for this day
                ts = day_obj.get_ts(length = ntimes_per_day)

                dat[iday,:] = ts
                                      

            bootstr = bootstrap(dat, n_samples = 999, method=2)
            bootstrap_list.append(bootstr)
            dat_list.append(dat)

        self.bootstrap_list = bootstrap_list
        self.dat_list = dat_list
        return bootstrap_list, dat_list
    


    ####################################
    def bootstrap_weekdays(self):
        """Do a bootstrap sampling of all weekdays (Mon-Fri)

        """

        # Find weekdays in the timeline
        indstart, indend = self.timeline.find_weekdays()

        ndays = len(indstart)
        ntimes_per_day = indend[0] - indstart[0]

        dat = numpy.zeros((ndays, ntimes_per_day))
        for iday in range(ndays):
            dat[iday,:] = self.ts_norm[indstart[iday]:indend[iday]]
        bootstr = bootstrap(dat,n_samples= 999, method=2)

        self.bootstr = bootstr
        return bootstr


    ####################################
    def plot_bootstrap_day(self, day_index, do_percentiles=True,
                           do_show=True):
        """Make a plot of the bootstrap data for 1 day-of-the-week
        """
        if self.bootstrap_list is None:
            self.bootstrap_day_of_week()
            
        boot = self.bootstrap_list[day_index]
        wbad = boot < 0.0
        boot[wbad] = numpy.nan
        pyplot.figure()
        ax = self.plot_bootstrap_as_2dhist(boot)
        
        # Get the x-axis values appropriate for the plot
        xrange = ax.xaxis.get_data_interval()
        nt = len(boot[0,:])
        xx = numpy.arange(nt) / float(nt) * \
            (xrange[1]-xrange[0]) + xrange[0]

        # overplot percentiles
        if do_percentiles:
            my_percentiles = [10,50,90]
            perc = numpy.nanpercentile(boot, my_percentiles, axis=0)
            nfilter = 100
            for pp in perc:
#                 pyplot.plot(filters.gaussian_filter1d(pp,nfilter),
#                             'blue',linewidth=2.5)
                pyplot.plot(xx, 
                            filters.median_filter(pp,nfilter)*
                            self.basic_data['dockcount'],
                            'blue',linewidth=2.5)

        # Label the plot with the correct day of week
        days_of_week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        pyplot.title("Station %d -- %s" % 
                     (self.ID, days_of_week[day_index]) )


        if do_show:
            pyplot.show()

        return ax, xx, perc

        
    ##########################
    def plot_bootstrap_as_2dhist(self, boot, norm_columns=False,
                                 nbins=[60,25], xtick_type='hours'):
        """Make a plot of bikes_available (actually ts_norm) vs.
        time with a 2D histogram showing the frequency at which 
        each time/bike combination occurs.

        norm_columns: normalize each column of the 2d hist so 
        that the peak value is 1.0
        """
        xpoints = numpy.indices(numpy.shape(boot))[1].ravel()
        ypoints = boot.ravel()
        wgood = ~numpy.isnan(ypoints)
        xpoints = xpoints[wgood]; ypoints=ypoints[wgood]

        # Add some noise to y positions to make the 2D histogram look better
        ypoints = ypoints + \
            numpy.random.randn(len(ypoints)) / \
            (self.basic_data['dockcount'] * 3.0)
        
        # Define number of bins in x and y
##        nbins = [60, 25]
        
        # Define the x- and y-axis value ranges
        ranges = numpy.array([[0, numpy.max(xpoints)+1], [0,1.]])
        
        # Get the 2D histogram data from the xpoints and ypoints
        hist,xedge,yedge = numpy.histogram2d(
            xpoints, ypoints,bins=nbins,
            range=ranges)

        # Normalize so that each column in the image has the same peak value
        if norm_columns:
            for icol in range(nbins[0]):
                hist[icol,:] = hist[icol,:] / numpy.max(hist[icol,:])

        # Locate the ticks for hours on the plot
        loc_hours = mdates.HourLocator(byhour = [3,6,9,12,15,18,21],
                                       tz = self.timeline.timezone)
        loc_allhours = mdates.HourLocator(tz = self.timeline.timezone)
        hrFmt = mdates.DateFormatter("%H:%M", tz=self.timeline.timezone)

        loc_days = mdates.WeekdayLocator(byweekday=range(7),
                                         tz = self.timeline.timezone)
        dayFmt = mdates.DateFormatter("%a")

        # Find all the datetime64's for a single day.  It doesn't have
        # to be this particular day.
        if xtick_type == 'hours':
#             ind_start = self.timeline.ind_daystart[1]
#             ind_end = self.timeline.ind_matched_dayend[1]
            ind_start = self.get_days(0)[0].ind_start
            ind_end = self.get_days(0)[0].ind_end
##            print 'indindindindind', ind_start, ind_end
        elif xtick_type == 'days':
            first_monday = self.get_days(0)[0]
            sundays = self.get_days(6)
            first_sunday = sundays[0]
            i = 1
            while first_sunday.date < first_monday.date:
                first_sunday = sundays[i]
                i += 1
            ind_start = first_monday.ind_start
            ind_end = first_sunday.ind_end
            pass
##        print 'BLOOOBA', ind_start, ind_end
        dtimes = self.timeline.datetimes[ind_start:ind_end]
        dtimes_num = mdates.date2num(dtimes)

        # Plot the 2D histogram as an image
        imdat = numpy.log10(hist.transpose())

        # Determine the xrange and yrange of the image in data units.
        # For the x-axis (extent[0:2]), we will use times.
#         my_extent = [0,0,0,0]
#         my_extent[0] = dtimes[0]
#         my_extent[1] = dtimes[1]
        my_extent = ranges.ravel()
        my_extent[0:2] = [numpy.min(dtimes_num),
                          numpy.max(dtimes_num)]
        my_extent[2:] = [0, self.basic_data['dockcount']]

##        print "EXTE", my_extent
        
        interp = 'gaussian'
##        interp = 'none'
        implot = pyplot.imshow(imdat,interpolation=interp,origin='lower',
                               extent=my_extent, aspect='auto')
        implot.set_cmap("Greys")

        # Set up time axis (on x-axis)
        ax = pyplot.gca()
        if xtick_type == 'hours':
            ax.xaxis.set_major_locator(loc_hours)
            ax.xaxis.set_major_formatter(hrFmt)        
#            ax.xaxis.set_minor_locator(loc_allhours)
        elif xtick_type == 'days':
            ax.xaxis.set_major_locator(loc_days)
            ax.xaxis.set_major_formatter(dayFmt)

            ax.xaxis.set_minor_locator(loc_hours)
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel('Bikes available')
        return ax


    ####################################
    def plot_bootstrap_week(self, do_percentiles=True,
                            do_show=True):
        """Plot a week's worth of bootstrap
        """
        if self.bootstrap_list is None:
            self.bootstrap_day_of_week()
        
        # Create a big bootstrap array with week-long time series
        nsample, ntimes_per_day = numpy.shape(self.bootstrap_list[0])
        seven = 7  # days per week
        boot = numpy.zeros((nsample, ntimes_per_day * seven))
        for ii, dayboot in enumerate(self.bootstrap_list):
            boot[:, ii*ntimes_per_day:(ii+1)*ntimes_per_day] = dayboot
            pass

        wbad = boot < 0.0
        boot[wbad] = numpy.nan

        fig = pyplot.figure(figsize=(12,4))
        ax = self.plot_bootstrap_as_2dhist(boot, nbins=[60*7,25],
                                           xtick_type='days')

        # Get the x-axis values appropriate for the plot
        xrange = ax.xaxis.get_data_interval()
        nt = len(boot[0,:])
        xx = numpy.arange(nt) / float(nt) * \
            (xrange[1]-xrange[0]) + xrange[0]

        if do_percentiles:
            my_percentiles = [10,50,90]
            perc = numpy.nanpercentile(boot, my_percentiles, axis=0)
            nfilter = 100
            for pp in perc:
                pyplot.plot(xx, 
                            filters.median_filter(pp,nfilter)*
                            self.basic_data['dockcount'],
                            'blue',linewidth=2.5)

        pyplot.title("Station %d" % (self.ID) )
        pyplot.subplots_adjust(bottom=0.15)
        if do_show:
            pyplot.show()

    ####################################
    def plot_double_boot(self, day_index=0):
        """Plot first and second half bootstraps for this station.
        """
        ndates = len(self.timeline.datetimes)
        middle_date = self.timeline.datetimes[ndates/2].date()

        # Run bootstrap on first half of data
        self.bootstrap_day_of_week(enddate=middle_date)

        # Plot first half
        pyplot.subplot(211)
        ax, xx, perc = self.plot_bootstrap_day(day_index, do_show=False)

        # Get percentile information from first half of data.  This
        # will be overplotted on the second plot.
        
        
        # Run bootstrap on second half of data
        self.bootstrap_day_of_week(startdate=middle_date)

        # Plot second half
        pyplot.subplot(212)
        self.plot_bootstrap_day(day_index, do_show = False)

        # Overplot percentiles from top plot
        for pp in perc:
            pyplot.plot(xx, filters.median_filter(pp,100)*
                        self.basic_data['dockcount'], '--y')

        fig = pyplot.gcf()
        fig.subplots_adjust(hspace=0.5)
        pyplot.show()

    
    ####################################
    def plot_day_stack(self, day_index):
        """Plot all Mondays on the same plot for this station
        """
        days_of_week = numpy.array(
            ['Mon','Tues','Wednes','Thurs','Fri','Satur','Sun'])
        daystring = days_of_week[day_index] + 'days'

        ind_start = self.timeline.ind_daystart[1]
        ind_end = self.timeline.ind_matched_dayend[1]
        dtimes = self.timeline.datetimes[ind_start:ind_end]
#         dtimes_num = mdates.date2num(dtimes)
        xx = dtimes

        pyplot.ion()
        fig = pyplot.figure(figsize=(5,10.))

        days = self.get_days(day_index)
        for ii, day in enumerate(days):
            pyplot.plot(dtimes, day.get_ts()+ii)
            
        # Locate the ticks for hours on the plot
        loc_hours = mdates.HourLocator(byhour = [3,6,9,12,15,18,21],
                                       tz = self.timeline.timezone)
        loc_allhours = mdates.HourLocator(tz = self.timeline.timezone)
        hrFmt = mdates.DateFormatter("%H:%M", tz=self.timeline.timezone)
        ax = pyplot.gca()
        ax.xaxis.set_major_locator(loc_hours)
        ax.xaxis.set_major_formatter(hrFmt)

        pyplot.title("Station %d -- All %s" % (self.ID, daystring))
        pyplot.xlabel("Time of day")
        pyplot.ylabel("Fraction of bikes + offset")
##        pyplot.show()
        raw_input("press enter to highlight a single time:")

        #-----------------------
        # Plot a rectangle highlighting one column of data
        time_ind = 500   # Choose which time to highlight
        r_width = 0.02
        r_height = 27.
        x_rect = [mdates.date2num(xx[time_ind])-0.5*r_width, 0.2]
###        print x_rect
        rect = patches.Rectangle( 
            x_rect, r_width, r_height,
            fill=False, color='green', linewidth=1.5)
        ax.add_patch(rect)
##        ax.apply_aspect()
        pyplot.draw()
        pyplot.show()
        raw_input("press enter:")

        #------------------------
        # Make a new figure showing 
        fig = pyplot.figure()
        dat = (self.dat_list[day_index])[:, time_ind]
        pyplot.subplot(211)
        pyplot.plot(dat)
        pyplot.xlabel("Week number")
        pyplot.ylabel("Frac. of bikes avail. at "+
                      xx[time_ind].strftime("%H:%M"))

        pyplot.subplot(212)
        acf = stats.acf(dat, fft=True, nlags=len(dat))
        pyplot.plot(acf)
        sig_acf_rand = numpy.std( stats.acf(numpy.random.randn(len(dat))))
        pyplot.hlines(0,0,25)
        pyplot.hlines([-sig_acf_rand, sig_acf_rand],0,25, linestyle='--')
        pyplot.xlabel("Lag")
        pyplot.ylabel("Autocorrelation function")
        pyplot.subplots_adjust(hspace=0.5)
        pyplot.show()

    ####################################
    def set_cluster_id(self, id):
        self.cluster_id = id
        return id

    ####################################
    def find_friends(self, linking_length, cluster_ID):
        """Find "friends" in a friends-of-friends algorithm.

        Also does a recursive call to itself to find friends-of-friends
        
        Assumes neighbors have been found (and therefore distances to 
        all other stations have been measured)
        """
        # Set this station's cluster ID
        self.set_cluster_id(cluster_ID)

###        print 'made', cluster_ID

        # Loop through all neighbor stations
        for i, neighbor_station in enumerate(self.neighbor_list):
            # Ignore stations separated by zero distance (i.e. itself!)
            if self.neighbor_dist[i] == 0:
                continue

            # If this neighbor is within linking length, then assign
            # it this cluster ID and find its neighbors
            if self.neighbor_dist[i] < linking_length:
                # Ignore stations already assigned a cluster
                if neighbor_station.cluster_id > 0:
                    continue

###                print "."

                # If this neighbor station is not assigned to a cluster, 
                # we have to go find its friends too.
                neighbor_station.find_friends(linking_length, cluster_ID)

            else:
                break  # Assumes neighbor_dist is sorted!

        return 0


    ####################################
    def normalize_features(self, features):
        """Normalize features to fall between 0 and 1.0
        """

        nfeat = len(features[0,:])
        norm_features = numpy.zeros(numpy.shape(features), dtype='float')
        feature_norm_params = numpy.zeros((nfeat, 2))
        
        # Loop over features
        for ii in range(nfeat):
            # Start by finding the normalization parameters, either by
            # using saved values or calculating new values
            if self.feature_norm_params is None:
                # Find the minimum value and the range of values
                minfeat = numpy.min(features[:,ii])
                feature_range = numpy.max(features[:,ii]) - minfeat
            else:
                minfeat = self.feature_norm_params[ii,0]
                feature_range = self.feature_norm_params[ii,1]
                
            # actually normalize the features
            norm_features[:,ii] = (features[:,ii] - minfeat) / \
                feature_range
            
            # Save the feature normalization parameters for later
            feature_norm_params[ii,:] = [minfeat, feature_range]

        # Save parameters for later (so we can un-normalize them)
        if self.feature_norm_params is None:
            self.feature_norm_params = feature_norm_params

        return norm_features 

    ####################################
    def regress(self, 
                C_grid = [0.1, 1.0, 10.0, 100.0, 1000.0],
                gamma_grid=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                subsample_size=3000,
                verbose = True, selection_criterion='training'):
        """Perform regression on the bike data at this station

        If you don't want to test a bunch of values, just set
        C_grid and gamma_grid to single-element arrays (or lists).
        """
        # Retrieve training data and cross-validation data
        self.setup_datasets(subsample_size)
        Xtrain, ytrain = self.get_training_data()
        Xcrossval, ycrossval = self.get_cross_validation_data()

        # Normalize the features (responses "y" should already be 
        # normalized since we use the normalized timeseries).
        Xtrain = self.normalize_features(Xtrain)
        Xcrossval = self.normalize_features(Xcrossval)

        # Create arrays to hold the error as a function of C and gamma.
        # Cross-validation error and training error.
        errorarr = numpy.zeros((len(C_grid), len(gamma_grid)) )
        error_train = numpy.zeros((len(C_grid), len(gamma_grid)) )

        print ("Station %d: Training SVR on grid of C and gamma "+\
            "parameters with "+\
            "%d examples") % (self.ID, subsample_size)

        for iC, my_C in enumerate(C_grid):
            for iG, gamma in enumerate(gamma_grid):
                # Fit the training data
                if verbose is True:
                    print "C, gamma, CV error, training error:", \
                        my_C, gamma,"...", 
                    sys.stdout.flush()

                my_epsilon = 0.01  # Default is 0.1
                svr_rbf = SVR(kernel='rbf', C = my_C, gamma=gamma,
                              cache_size = 500, epsilon=my_epsilon)
                fit = svr_rbf.fit(Xtrain, ytrain)

                if verbose is True:
                    print ".",
                    sys.stdout.flush()
##                print "C=", fit.get_params()['C']

                # Make predictions for the cross validation data and
                # calculate the error in those predictions.
                result_CV = fit.predict(Xcrossval)
                errorarr[iC, iG] = self.error_crossval(result_CV, ycrossval)

                result_train = fit.predict(Xtrain)
                error_train[iC,iG] = self.error_crossval(result_train, ytrain)

                if verbose is True:
                    print errorarr[iC,iG], error_train[iC,iG]

                pass
            pass
        
        ### Now we must choose the "best" combination of C and gamma.
        # Normally one would use the the values that give the lowest 
        # cross-validation error, but it's too expensive to train SVMs 
        # with a large fraction of the dataset for many values of C and
        # gamma.  Instead we want to get small values for the training
        # error and small values for the C parameter.  
        #    The training error
        # (for fixed C and gamma) changes little with increasing sample
        # size, whereas the cross-val error converges toward the training 
        # error as we increase sample size.  So we can choose the 
        # lowest training error now (which we find using small sample size),
        # then when we increase the sample size for our final result
        # the cross-val error will be smaller.
        #    It also turns out the training error varies only slowly with
        # C.  Furthermore, the SVR trains much faster for small values of
        # C.  So we can choose a small value for C as long as the resulting
        # error is within 10% of the "best" training error, and then
        # it will enable us to train on larger sample sizes in a reasonable
        # amount of time, and we will end up with lower cross-val error.

        # Find the position of the minimum CV and training error
#         iC, iG = numpy.unravel_index( numpy.argmin(errorarr),
#                                       numpy.shape(errorarr))
        if selection_criterion == 'training':
            minerr = numpy.min(error_train)
            iC, iG = numpy.unravel_index( numpy.argmin(error_train),
                                          numpy.shape(error_train) )
            best_gamma = gamma_grid[iG]

            # Find values of C that give an error close to the minimum error.
            w_smallerr = error_train[:, iG] < minerr*1.10

            # Choose the smallest acceptable C
            best_C = numpy.min(numpy.array(C_grid)[w_smallerr])
        elif selection_criterion == 'crossval':
            iC, iG = numpy.unravel_index( numpy.argmin(errorarr),
                                          numpy.shape(errorarr) )
            best_C = C_grid[iC]
            best_gamma = gamma_grid[iG]
        else:
            raise Exception("selection_criterion neither training "+
                            "nor crossval")

        # If there is more than one combination of (C,gamma), then 
        # Select the best one and re-run the fit with that best combo.  
        # If there's only 1 (C,gamma) provided, then we don't need
        # to run the fit again.
        if len(C_grid) > 1 or len(gamma_grid) > 1:
            print "Best parameters C, gamma:", best_C, best_gamma

##            print "-----", fit.get_params()['C'], fit.get_params()['gamma']

            # Run the regression (again) with the best parameters, if needed
            if ( (fit.get_params()['C'] != best_C) or 
                 (fit.get_params()['gamma'] != best_gamma) ):
                print "Re-training w/ best values"
                svr_rbf = SVR(kernel='rbf', C = best_C, gamma = best_gamma,
                              cache_size=500, epsilon=my_epsilon)
                fit = svr_rbf.fit(Xtrain, ytrain)
            
        # save the result
        self.svr_fit = fit
        self.minerr = (numpy.min(error_train), numpy.min(errorarr))

    ####################################
    def get_training_data(self):
        """Retrieve training data for this station
        """
        if self.training_ind is None:
            self.setup_datasets()

        # Get responses (the number-of-bikes info)
        ytrain = self.ts_norm[self.training_ind]
        
        # Get features (mostly time-of-day, day-of-week, etc.)
        Xtrain = self.get_features()[self.training_ind]

        return Xtrain, ytrain

    ####################################
    def get_cross_validation_data(self):
        """Retrieve cross validation data for this station
        """
        if self.cross_validation_ind is None:
            self.setup_datasets()

        ycrossval = self.ts_norm[self.cross_validation_ind]

        Xcrossval = self.get_features()[self.cross_validation_ind]
        return Xcrossval, ycrossval

    ####################################
    def get_test_data(self):
        """Retrieve test data for this station
        """
        if self.test_ind is None:
            self.setup_datasets()

        ytest = self.ts_norm[self.test_ind]

        Xtest = self.get_features()[self.test_ind]
        return Xtest, ytest

    ####################################
    def setup_datasets(self, subsample_size=1000, 
                       method = 3):
        """From the base data, create training, cross-validation, and test data
        sets.

        This actually just specifies the index values into the original 
        arrays that will achieve the appropriate sub-samples.

        subsample_size: If we use all the data then the arrays are too
          big for SVM algorithm to work.  So we can choose a smaller
          sub-sample to make the SVM feasible.

        method: There are 2 ways to determine the cross-validation
          (and test) data sets.  (1) select a data subsample, and split it
          up into training, cross-val, and test data sets with a 60-20-20 % 
          split.  (2) If subsample_size is much smaller than the original
          data size, then we can use "discarded" values as the cross-validation
          and test data sets.  In this case, we just take the "discarded" 
          data and split it in half (half cross-val, half test data).  Setting
          method=2 chooses case (2).
          (3) Means we split the data into first and second halves,
          then draw training from the first half and crossval from the second.
        """

        # Exclude "bad" data points (where no data was available)
        ind_good = (numpy.where(self.ts_norm >= 0.0))[0]

###        if large_crossval:
        if method == 2:
            # Use a small sampling of the total data for the training set,
            # but a large sampling for both the crossval and test sets.
            numpy.random.shuffle(ind_good)
            ind_all = ind_good
            ntraining = subsample_size

            # nCV = number of cross-validation examples = half of the
            # remaining points.
            nCV = round( (len(ind_good) - subsample_size) / 2.0 )
            
        elif method == 1:
            # Use only a random subsample of the total data, otherwise the
            # dataset is too large to run SVM.  This gives small numbers
            # of training examples as well as crossval and test examples.
            ind_all = numpy.random.choice(ind_good, size=subsample_size, 
                                          replace=False)

            # Split sample into different components.
            frac_training = 0.60
            frac_CV = 0.20
            frac_test = 0.20  # Not actually used
        
            ntraining = round(frac_training * subsample_size)
            nCV = round(frac_CV * subsample_size)

        elif method == 3:
            # Choose a time-segregated data sets.  The training set is a 
            # small subsample drawn from the first half of the data, while
            # crossval and test samples are drawn from the second half.
            half = len(ind_good) / 2
            ind_firsthalf = ind_good[0:half]
            ind_secondhalf = ind_good[half:]
            
            ntraining = subsample_size
            numpy.random.shuffle(ind_firsthalf)
            self.training_ind = ind_firsthalf[0:ntraining]
            
            nCV = round(half / 2.0)
            numpy.random.shuffle(ind_secondhalf)
            self.cross_validation_ind = ind_secondhalf[0:nCV]
            self.test_ind = ind_secondhalf[nCV:]

        if method < 3:
            self.training_ind = ind_all[0:ntraining]
            self.cross_validation_ind = ind_all[ntraining:ntraining+nCV]
            self.test_ind = ind_all[nCV+ntraining:]

#         print "Len of datasets:", len(self.training_ind),\
#             len(self.cross_validation_ind), len(self.test_ind)
#         print self.cross_validation_ind
        return 0


    ####################################
    def get_features(self):
        """Retrieve desired features 'X' for use with regression

        This includes all datapoints (we haven't yet split the sample
        into subsamples for training, cross-validation, etc.).
        """
        extra_features = self.extra_features

        # Get timeline features
        # (time-of-day, day-of-week, week-of-year...)
        features = self.timeline.get_features()

        # Select only a subset of features.  Always keep first 2 features.
        ind = [0,1]
        if extra_features is not None:
            if 'week_of_year' in extra_features:
                ind.append(2)
            if 'holiday' in extra_features:
                ind.append(3)
            if 'sunrise' in extra_features:
                ind.append(4)

        features = features[:,ind]

        allfeat = features

        # Other features?
        if extra_features is not None:
            if 'weather' in extra_features:
                wfeat = self.weather.get_weather_features()
                allfeat = numpy.hstack((features, wfeat))
        
        return allfeat


    ####################################
    def error_crossval(self, predictions, observations):
        """Calculate the error between the predicted values and crossval data
        
        Use mean-squared error method.
        Returns a scalar value giving the error.
        """
        NN = len(observations)
        if NN != numpy.size(predictions):
            raise Exception("mismatch in number of elements")
        err = numpy.sum((predictions - observations)**2.0) / NN
        return err


    ####################################
    def learning_curves(self):
        """Create "learning curves" for SVM regression fits

        A learning curve is a plot of error vs. number of training examples
        included in the training set.  It shows both training error and
        cross-validation error, which should converge.
        """
        n_example = [10, 30, 100, 300, 1000, 3000, 10000]
        
        error_CV = numpy.zeros(len(n_example), 'float')
        error_train = numpy.zeros(len(n_example), 'float')
        for ii, subsample_size in enumerate(n_example):
            print "Subsample_size:", subsample_size
            self.setup_datasets(subsample_size = subsample_size)
            Xtrain,ytrain = self.get_training_data()
            Xcv, ycv = self.get_cross_validation_data()

            Xtrain = self.normalize_features(Xtrain)
            Xcv = self.normalize_features(Xcv)
            
            fit = SVR(kernel='rbf', C = 100.0, gamma = 100.0, 
                      cache_size=500).fit(Xtrain, ytrain)

            result_cv = fit.predict(Xcv)
            result_tr = fit.predict(Xtrain)
            
            error_CV[ii] = self.error_crossval(result_cv, ycv)
            error_train[ii] = self.error_crossval(result_tr, ytrain)
            print error_CV[ii], error_train[ii]

        pyplot.plot(n_example, error_train)
        pyplot.plot(n_example, error_CV)
        pyplot.show()

    ####################################
    def plot_prediction_comparison(self):
        """Compare svm predictions to real observations for this Station
        """
        # Get the predictions from our SVM model
        Xall = self.get_features()
        Xall = self.normalize_features(Xall)
        ypredict = self.svr_fit.predict(Xall)

        # This is only because of mdates
        fig, ax = pyplot.subplots()  # get access to both figure and axes

        # Create array of datetimes for the x-axis
        xx = numpy.array(self.timeline.datetimes)

        # Plot original data
        y = self.ts_norm
        pyplot.plot(xx, y)

        # Plot predictions
        pyplot.plot(xx, ypredict)

        # Create "locator objects" which indicate the positions
        # of months on the x-axis
        loc_months = mdates.MonthLocator(tz = self.timeline.timezone)
        loc_weekdays = mdates.WeekdayLocator(byweekday=range(7),
                                             tz=self.timeline.timezone)

        # Indicate months with 3-letter format (e.g. Aug), and
        # weekdays with both the day-of-month (e.g. 09) and the
        # day-of-week (e.g. 'Wed')
        monthFmt = mdates.DateFormatter("%b")
        weekdayFmt = mdates.DateFormatter("%d %a")

        # Set major ticks to indicate the Months. Increase the "pad"
        # between the axis and the tick labels to allow room for 
        # minor tick labels.
        ax.xaxis.set_tick_params(which='major', pad=50)
        ax.xaxis.set_major_locator(loc_months)
        ax.xaxis.set_major_formatter(monthFmt)

        # Set minor ticks on x-axis to indicate the day of the week/month
        ax.xaxis.set_minor_locator(loc_weekdays)
        ax.xaxis.set_minor_formatter(weekdayFmt)

        # Move up the bottom of the plot axes so that we have enough
        # room for the x-axis tick labels
        fig.subplots_adjust(bottom=0.2)

        # Rotate the minor tick labels to make more space between them
        labels = ax.get_xticklabels(minor=True)
        pyplot.setp(labels, rotation=60)

#         # set the xrange
#         datemin = datetime.date(.year, 1, 1)
#         datemax = datetime.date(r.date.max().year + 1, 1, 1)
#         ax.set_xlim(datemin, datemax)

        pyplot.show()

    ####################################
    def plot_prediction_boot(self):
        """Overplot svr prediction on top of bootstrap results
        """

        ### Get the predictions from our SVM model.
        # First get the feature set for all times.
        Xall = self.get_features()
        
        # Find times for just the first full week.
        first_monday = self.get_days(0)[0]
        sundays = self.get_days(6)
        first_sunday = sundays[0]
        i = 1
        while first_sunday.date < first_monday.date:
            first_sunday = sundays[i]
            i += 1
        ind_start = first_monday.ind_start
        ind_end = first_sunday.ind_end

        Xall = Xall[ind_start:ind_end, :]
        Xall = self.normalize_features(Xall)
        ypredict = self.svr_fit.predict(Xall)

        ### 
        # Plot the bootstrap for 1 week
        self.plot_bootstrap_week(do_percentiles=False, do_show=False)
        
        # Overplot the SVM prediction
        xx = self.timeline.datetimes[ind_start:ind_end]
        yy = ypredict * self.basic_data['dockcount']
        pyplot.plot(xx, yy, linewidth = 2.5)

        pyplot.show()

        pass


    ####################################
    def predict(self,timestring, fmt='%Y-%m-%d %H:%M:%S'):
        """
        """
        dt = datetime.datetime.strptime(timestring, fmt)

        # Get minute-of-day and day-of-week for this time
        DOW = dt.weekday()
        MOD = dt.minute + 60 * dt.hour

        days_of_week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

        print "Prediction for", days_of_week[DOW], timestring

        # minute of week serves as index to the bootstrap array
        ##MOW = DOW*1439 + MOD

        ind = MOD

        # Select columns from the bootstrap
        distribution = (self.bootstrap_list[DOW])[:, ind-3:ind+3]
        distribution *= self.basic_data['dockcount']

        # 
        pred_obj = PredictionObj(distribution.ravel(), 'bikes_available')
        return pred_obj

    ####################################
    def test_regress(self, myC=100., myGamma=100., subsample_size=1000):
        """Test whether careful choice of data subset gives better regression

        This is meant as a testing ground to figure out how the 
        cross-validation error changes with different sizes of subsample,
        and different ways of choosing the training set.

        I found that choosing random sub-samples based on derivatives
        does not improve the fit.  Increasing the size of the sub-sample
        does improve the fit.
        """
        
        # Get the original data
        Xall = self.timeline.get_features()
        yall = self.ts_norm

        # Choose a contiguous subset of data (e.g. 1 week)
        start = 10000; stop = 20000
        ysub = yall[start:stop]
        Xsub = Xall[start:stop, :]

        # Get rid of bad points
        wgood = ysub >= 0.0
        ysub = ysub[wgood]
        Xsub = Xsub[wgood,:]

        # normalize the features
        Xsub = self.normalize_features(Xsub)

        # draw random samples of the data
###        subsample_size = 1000
        ind_rand = numpy.random.choice(len(ysub), size=subsample_size,
                                       replace=False)
        

        # look at derivatives to help choose data subset
        delta = 4    # half-width of window over which to get derivative
        diff = numpy.roll(ysub,-1*delta) - numpy.roll(ysub,1*delta)
        deriv = (diff / (2*delta))

#         pyplot.subplot(211)
#         pyplot.plot(ysub)
#         pyplot.subplot(212)
#         pyplot.plot(numpy.abs(deriv))
##        pyplot.show()

##        return deriv, ysub

        # Choose subset of the data with special weights based on
        # the derivative.
        prob = 1 + (numpy.abs(deriv) / numpy.std(deriv))**2.0
##        print prob
        prob = prob / numpy.sum(prob)  # normalize so sum=1
        ind_choice = numpy.random.choice(len(ysub), size=subsample_size, 
                                         replace = False, p = prob)

        # Plot positions of chosen sample points
#         pyplot.plot(ind_rand, numpy.full(subsample_size,0.05), '.')
#         pyplot.plot(ind_choice, numpy.full(subsample_size,0.07), '+')
#         pyplot.show()

        ### Create training and cross-validation samples
        ntrain = numpy.round(0.8 * subsample_size)

        # for the totally random sample
        ind_train_r = ind_rand[0:ntrain]
        ind_val_r = ind_rand[ntrain:]
        ytrain_r = ysub[ind_train_r]
        Xtrain_r = Xsub[ind_train_r]
        yval_r = ysub[ind_val_r]
        Xval_r = Xsub[ind_val_r]

        # for the specially chosen sample
        ind_train_c = ind_choice[0:ntrain]
        ind_val_c = ind_choice[ntrain:]
        ytrain_c = ysub[ind_train_c]
        Xtrain_c = Xsub[ind_train_c]
        yval_c = ysub[ind_val_c]
        Xval_c = Xsub[ind_val_c]
        
        # Create another cross-validation sample using all points 
        # in the time-series (not just the subsample points)
        yval_all = ysub
        Xval_all = Xsub

        # Train the SVRs
        print "Training..."
#         myC = 100.
#         myGamma = 100.
        svr_r = SVR(kernel='rbf', C = myC, gamma=myGamma,cache_size=500)
        fit_r = svr_r.fit(Xtrain_r, ytrain_r)
        svr_c = SVR(kernel='rbf', C = myC, gamma=myGamma,cache_size=500)
        fit_c = svr_c.fit(Xtrain_c, ytrain_c)

        # Check cross-validation error
        result_r = fit_r.predict(Xval_r)
        result_c = fit_c.predict(Xval_c)
        error_r = self.error_crossval(result_r, yval_r)
        error_c = self.error_crossval(result_c, yval_c)

        # Check error on excluded data
        result_all_r = fit_r.predict(Xval_all)
        result_all_c = fit_c.predict(Xval_all)
        error_all_r = self.error_crossval(result_all_r, yval_all)
        error_all_c = self.error_crossval(result_all_c, yval_all)

        print "errors"
        print error_r, error_c, error_all_r, error_all_c
        return 0

####################################
####################################
class TimeSegment:
    """Class to hold a sub-segment of time-series data"""

    ####################################
    def __init__(self, ind_start, ind_end, station, ID = -1):
        """Create a TimeSegment (sub- time series) object.

        ind_start: Index to the timeline indicating the start of the day
        station: "parent" station object, which must contain a timeline
        ID: An ID number to give the instance.  It can be any
           arbitrary number.
        """
#         # Determine ind_end, the last index corresponding to this day
#         ind_end = station.timeline.find_dayend(ind_start = ind_start)

        self.ind_start = ind_start
        self.ind_end = ind_end

        self.ID = ID

        # point to time series data
        self.table = station.table[ind_start:ind_end]
        self.station = station
        self.station_id = station.ID
        self.timeline = station.timeline

        # Define the default time series, which is used for comparison
        # with other TimeSegment objects
        if 'ts_norm' in self.table.keys():
            self.ts = self.table['ts_norm']
#         if 'ts_clean' in self.table.keys():
#             self.ts = self.table['ts_clean']
        else:
            self.ts = self.table['bikes_available']
        pass

    ####################################
    def distance(self, other_time_segment):
        """Find the distance between this TimeSegment and another, where distance
        is defined using a given metric, e.g. the Euclidean distance between
        their time series data, the physical distance on the surface of the 
        Earth, or some weighted combination of the two.

        other_time_segment: a different TimeSegment object
        """
        # Create TimeSeries object for each segment
        ts1 = TimeSeries(self.ts)
        ts2 = TimeSeries(other_time_segment.ts)

        # Measure Euclidean distance between these days
        euclid_dist = ts1.euclidean_distance(ts2)

        # Use other measures like weather...

        # Use weightings among the different dimensions

        # Determine the final distance
        dist = euclid_dist

        return dist

    ####################################
    def find_neighbors(self, other_TS_list=None,
                       n_neighbors = 100):
        """Find other time segments (possibly from other stations) that are
        nearest neighbors to this day."""        

        # Loop over candidate neighbors, measuring the 'distance' to each
        dist_all = []
        for other_TS in other_TS_list:
            dist = self.distance(other_TS)
            dist_all.append(dist)

        # Now sort by distance
        dist_arr = numpy.array(dist_all)
        ss = numpy.argsort(dist_arr)
        neighbor_list = []
        for ii in range(len(dist_all)):
            neighbor_list.append( other_TS_list[ss[ii]])

#         print "FIND NEIGHBORS"
#         print len(dist_all)
#         print dist_arr[ss]
#         print len(neighbor_list)

        # Cut down from all neighbors to just the nearest ones(?).
        self.neighbor_dist = dist_arr[ss][0:n_neighbors]
        self.neighbor_list = neighbor_list[0:n_neighbors]
        
        pass

    ####################################
    def plot_ts(self):
        """Plot various versions of the time series"""
        xx = self.timeline.times[self.ind_start:self.ind_end]
        yy = self.table['ts_norm']
        pyplot.plot(xx, yy)

        yy = self.table['ts_clean']
        pyplot.plot(xx,yy)

        pyplot.show()

    

####################################
####################################
class DayTS(TimeSegment):
    """Class to hold a day's worth of time series data"""

    ####################################
    def __init__(self, ind_start, station, ID = -1):
        """Create a DayTS (day time series) object.

        ind_start: Index to the timeline indicating the start of the day
        station: "parent" station object, which must contain a timeline
        ID: An ID number to give the day instance.  It can be any
           arbitrary number.
        """
        # Determine ind_end, the last index corresponding to this day
##        ind_end = station.timeline.find_dayend(ind_start = ind_start)
        wstart = station.timeline.ind_daystart == ind_start
        ind_end = station.timeline.ind_matched_dayend[wstart]

        TimeSegment.__init__(self, ind_start, ind_end, station, ID)

        first_dt = self.timeline.datetimes[self.ind_start]
        self.day_of_week_index = first_dt.weekday()
        self.date = first_dt.date()
        pass

    ####################################
    def find_neighbors(self, other_day_list = None, station_list=None,
                       n_neighbors = 100):
        """Find neighbor days.
        
        NB: This is an extension the TimeSegment.find_neighbors() 
        method
        """
        
        # If station_list is provided, then create a list of all possible
        # similar objects to compare against.
        if station_list is not None:
            other_day_list = []
            for station in station_list:
                for day in station.daylist:
                    other_day_list.append(day)

        TimeSegment.find_neighbors(self, other_day_list)

        pass

    ####################################
    def get_ts(self, length=None):
        """Return (a pointer to) the timeseries for this day
        """
        if length is None:
            return self.ts
        else:
            # If it's already the right length, just return it.
            if len(self.ts) == length:
                return self.ts

            elif len(self.ts) < length:
                # If it's not the right length, we have to create a new,
                # padded array.
                
                # Create an array giving the minute-of-day for each time
                min_of_day = numpy.arange(len(self.ts),dtype='int')
                my_datetimes = self.timeline.datetimes[self.ind_start:\
                                                           self.ind_end]
                for ii, dt in enumerate(my_datetimes):
                    min_of_day[ii] = dt.minute + dt.hour*60

                # desired_arr = numpy.arange(length, dtype='int')
                
                # Create a mask array that will have NANs where data is missing
                # and 1.0 elsewhere
                maskarr = numpy.zeros(length) * numpy.nan
                maskarr[min_of_day] = self.ts
                
                return maskarr
                    

            elif len(self.ts) > length:
                raise Exception("Can't deal with days >1440 minutes!")
                
        
####################################
####################################
class Timeline:
    """Class to manipulate timelines"""
    def __init__(self, timearr, timezone = None, 
                 load_filename=None):
        """Create a timeline object.

        timearr: array of numpy.datetime64 objects.
        load_filename: if provided, then read Timeline data from the 
          specified pickle file rather than creating new data.  If set,
          then we ignore 'timearr' argument.
        """
##        self.holidays = None
        self.read_holidays()
        self.read_sunrise()

        if load_filename is not None:
            self.load_data(load_filename)
            return

        ### Below occurs only if no loadfile is provided
        self.times = timearr
        self.timezone = timezone

        self.datetimes = None
        self.ind_daystart = None
        self.ind_dayend = None
        self.ind_matched_dayend = None
        self.tnorm = None
        self.features = None

        pass

    ####################################
    def save_data(self, save_filename):
        """Save the underlying timeline data
        
        """
        print "Saving timeline data to file ", save_filename

        file = open(save_filename,'wb')

        pickle.dump(self.times, file, -1)
        pickle.dump(self.timezone, file, -1)
        pickle.dump(self.datetimes, file, -1)
        pickle.dump(self.ind_daystart, file, -1)
        pickle.dump(self.ind_dayend, file, -1)
        pickle.dump(self.ind_matched_dayend, file, -1)
        pickle.dump(self.tnorm, file, -1)
        pickle.dump(self.features, file, -1)
                    
        file.close()

    ####################################
    def load_data(self, filename):
        print "Loading timeline data from file", filename

        file = open(filename, 'rb')
        
        self.times = pickle.load(file)
        self.timezone = pickle.load(file)
        self.datetimes = pickle.load(file)
        self.ind_daystart = pickle.load(file)
        self.ind_dayend = pickle.load(file)
        self.ind_matched_dayend = pickle.load(file)
        self.tnorm = pickle.load(file)
        self.features = pickle.load(file)

        file.close()


    ####################################
    def make_datetimes(self):
        """Convert the numpy.datetime64 array into an 
        array of Python datetime objects

        """
        self.datetimes = []
        for time in self.times:
            dt = datetime.datetime.fromtimestamp(time.astype('int'), 
                                                 self.timezone)
            self.datetimes.append(dt)
        pass

    ####################################
    def find_all_daystarts(self):
        """Find each point in the timeline that represents the start
        of a new calendar day.  Returns the array of index values
        pointing to the day starts.  
        
        Also find w_dayend, the end-of-day equivalent.
        """
        # Create array to hold the results
        w_daystart = numpy.zeros(len(self.datetimes), dtype='bool')
        w_dayend = numpy.zeros(len(self.datetimes), dtype='bool')
        for ii, dt in enumerate(self.datetimes):
            if dt.minute == 0 and dt.hour == 0:
                w_daystart[ii] = True
            if dt.minute == 59 and dt.hour == 23:
                w_dayend[ii] = True

        self.ind_daystart = numpy.where(w_daystart)[0]
        self.ind_dayend = numpy.where(w_dayend)[0]
        pass


    ####################################
    def match_day_startend(self):
        """Fix daystarts and dayends so that they correspond to the same day

        Make it so that dayend[0] is the last minute of the day starting
        at daystart[0]

        Returns new versions of ind_daystart and ind_dayend, but does not
        alter self.ind_daystart...
        """
        istart = self.ind_daystart
        iend = self.ind_dayend

        # Get rid of dayends at the start of the array that take place
        # before the first daystart
        while iend[0] < istart[0]:
            iend = iend[1:]
            pass

        # Get rid of daystarts at the end of the array that take place
        # after the last dayend.  This cuts off the last partial day.
        while istart[-1] > iend[-1]:
            istart = istart[0:-1]
            pass

        # Check that each dayend corresponds to the correct daystart
        for ii in range(len(istart)):
            if self.datetimes[istart[ii]].date() != \
                    self.datetimes[iend[ii]].date():
                print istart
                print iend
                raise Exception("Bad dayends/daystarts")

        return istart, iend

    ####################################
    def find_daystart(self, input_time):
        """Find the point in the timeline that represents the beginning
        of the calendar day in which input_time falls.
        
        input_time: a numpy.datetime64 object
        """
        # Make sure we have ind_daystart figured out for this Timeline
        if self.ind_daystart is None:
            self.find_all_daystarts()

        # Compare each daystart with the input time
        diffs = input_time - self.times[self.ind_daystart]
        w_pos = numpy.where(diffs > 0)[0]
        min_ind = numpy.argmin(diffs[w_pos])
        myind = self.ind_daystart[w_pos[min_ind]]
        
        return myind

    ####################################
    def find_dayend(self, input_time=None, ind_start = None,
                    approx = False):
        """Find the point in the timeline that represents the end
        of the calendar day in which input_time falls.
        
        input_time: a numpy.datetime64 object
        ind_start: As an alternative to providing the time, you can provide
           the index to a time in the timeline.  input_time must be None
           for this to work.
        approx: If the dayend corresponding to this time does
           not exist in the timeline, default behavior is return None.
           If approx == True, then just find the nearest value possible
        """
        #
        # Need to deal with case where time is outside the timeline!!!
        #

        if self.ind_dayend is None:
            self.find_all_daystarts()

        if input_time is None:
            if ind_start is None:
                raise Exception("Timeline.find_dayend() -- no inputs given!")
            else:
                input_time = self.times[ind_start]

        diffs = input_time - self.times[self.ind_dayend]
        w_neg = numpy.where(diffs < 0)[0]
        if len(w_neg) == 0:
            # The appropriate end-of-day is off the timeline!
            if approx is True:
                # Just return the last time of the timeline
                myind = len(self.times) - 1
            else:
                # Return None to indicate that we didn't find the dayend
                myind = None

        else:  # Normal behavior
            min_ind = numpy.argmin( numpy.abs(diffs[w_neg]))
            myind = self.ind_dayend[w_neg[min_ind]]

        
        return myind


    ####################################
    def match_dayends(self):
        """For each daystart, find the matching dayend
        """
        if self.ind_dayend is None:
            self.find_all_daystarts()
            pass
        wgood = self.ind_dayend > numpy.min(self.ind_daystart)
        matched_dayend = self.ind_dayend[wgood]

        # Usually the last day is a partial day, so we have only its
        # first minute, not its last one
        if len(self.ind_daystart) > len(matched_dayend):
            matched_dayend = numpy.append(matched_dayend, -1)

        # Check results
        for i, ind_end in enumerate(matched_dayend):
            ind_start = self.ind_daystart[i]
            if self.datetimes[ind_end].date() != \
                    self.datetimes[ind_start].date():
                raise Exception("Daystart/Dayend mismatch!")
            

        self.ind_matched_dayend = matched_dayend

    ####################################
    def tnorm_periodic(self, period=1440, t_first_element=0):
        """Convert to normalized periodic time.

        NB: One problem with this routine is that it doesn't account
        for daylight savings time!!!

        e.g.: convert each time in a timeline to the time-since-the-
        beginning-of-the-week.  
        
        period: period length, in minutes; e.g. 1 day = 1440
        t_first_element: The normalized time to assign to the first element
          of the timeline.  For example, if you have a 1-day period, and
          the first element of the timeline corresponds to 1:00am, you 
          might set t_first_element = 60.
        """
        ndat = len(self.times)
        nperiods = ndat / period + 1

        # Create an nperiods by period array.
        ygrid = numpy.mgrid[0:nperiods, 0:period][1]

        # Reshape the array to get a periodic array 
        time_per = numpy.reshape(ygrid, nperiods * period)
        
        # reduce to the correct number of elements
        time_per = time_per[0:ndat]
        
        # Shift index values as desired
        time_per = numpy.roll(time_per, -1*t_first_element)
        return time_per
        pass


    ####################################
    def tnorm_daily(self, set_tnorm=1):
        """Find the time normalized on a daily schedule

        """
        tnorm = numpy.zeros(len(self.datetimes))
        for i,dt in enumerate(self.datetimes):
            # hour of the day
            HOD = float(dt.hour)

            # minute of the hour
            MOH = float(dt.minute)

            tnorm[i] = \
                HOD * MINUTES_PER_HOUR + \
                MOH
                
        if set_tnorm > 0:
            self.tnorm = tnorm
        return tnorm
        pass
 
    ####################################
    def tnorm_weekly(self, set_tnorm=1):
        """Find the time normalized on a weekly schedule

        """
        tnorm = numpy.zeros(len(self.datetimes))
        for i,dt in enumerate(self.datetimes):
            # day of week
            DOW = float(dt.strftime("%w"))

            # hour of the day
            HOD = float(dt.hour)

            # minute of the hour
            MOH = float(dt.minute)

            tnorm[i] = \
                DOW * MINUTES_PER_DAY + \
                HOD * MINUTES_PER_HOUR + \
                MOH
                
        if set_tnorm > 0:
            self.tnorm = tnorm
        return tnorm
        pass


    ####################################
    def find_weekdays(self):
        """find the daystarts of all weekdays

        """

        ind_weekdays = numpy.array([], dtype='int')
        ind_weekday_ends = numpy.array([], dtype='int')

        # Loop through day index values 0 to 4 (==Monday to Friday).
        for day_index in range(5):
            ind_thisday, ind_thisday_end = \
                self.find_day_of_week(day_index)
            ind_weekdays = numpy.append(ind_weekdays, ind_thisday)
            ind_weekday_ends = numpy.append(
                ind_weekday_ends, ind_thisday_end)
            pass

        return ind_weekdays, ind_weekday_ends
    
    ####################################
    def find_day_of_week(self, day_index):
        """Find all Mondays (for example) in the timeline

        day_index: Monday is 0, Sunday is 6

        Assumes all day starts have been found already
        """
        ind_daystart, ind_dayend = self.match_day_startend()
        n_days = len(ind_daystart)
        w_thisday = numpy.zeros(n_days, dtype='bool')

        # Loop over all daystarts
        for ii in range(n_days):
            # Find the datetime for this day's first minute
            date_time = self.datetimes[ind_daystart[ii]]
            
            # Check if this is the correct day of the week
            if date_time.weekday() == day_index:
                w_thisday[ii] = True

        # Create index values indicating the locations of the correct
        # days.  These should be index values into the list
        # of datetimes (much as  self.ind_daystart is an index into
        # the list of datetimes)
        ind_thisday = ind_daystart[w_thisday]
        ind_thisday_end = ind_dayend[w_thisday]

        return ind_thisday, ind_thisday_end

    ####################################
    def make_features(self, extra_features=None):
        """Define some timeline features for regression methods

        return value: array with ntimes rows and N_FEATURES columns
        
        extra_features: list of feature names, among 'week_of_year',
          'holiday', 'sunrise'
        """
        N_FEATURES = 2
        if extra_features is not None:
            N_FEATURES += len(extra_features)

        ntimes = len(self.datetimes)

        # Create array to store the output.        
        result = numpy.zeros((ntimes, N_FEATURES))

        # Loop over all datetimes, converting each into a "feature"
        for ii, dt in enumerate(self.datetimes):
            jfeat = 0  # index to the number of features
            
            # minute of the day
            result[ii, jfeat] = dt.minute + 60 * dt.hour
            jfeat += 1

            # day of the week
            result[ii, jfeat] = dt.weekday()
            jfeat += 1

            if extra_features is not None:
                # week of the year
                if 'week_of_year' in extra_features:
                    year, weeknumber, daynumber = dt.isocalendar()
                    result[ii, jfeat] = weeknumber
                    jfeat += 1

                # Number of days to nearest holiday
                if 'holiday' in extra_features:
                    min_delta = 365
                    for holiday in self.holidays:
                        delta_t = holiday - dt
                        if abs(delta_t.days) < abs(min_delta):
                            min_delta = delta_t.days
                    result[ii, jfeat] = min_delta
                    jfeat += 1

                # Incorporate sunrise/sunset time.
                if 'sunrise' in extra_features:
                    result[ii,jfeat] = self.sunlight_index(dt)

                pass

        # get rid of extra features (columns) with all zeros.
        ind = []
        for ii in range(N_FEATURES):
            # Find columns with > 0 non-zero elements.
            if (result[:,ii] != 0.0).sum() > 0:
                ind.append(ii)
        result = result[:,ind]

        self.features = result
        return result
        
    ####################################
    def get_features(self):
        """ Retrieve Timeline features for use with regression
        """
        if self.features is None:
            self.make_features()
        return self.features


    ####################################
    def read_holidays(self):
        """ Read the dates of holidays from a file.
        
        Return a list of datetime objects holding the dates of
        the holidays.
        """
        filename = 'holidays.txt'
        datestrings = ascii.read(filename)['col1']

        format = '%Y-%m-%d'
        holiday_datetime_list = []
        for datestr in datestrings:
            dt = datetime.datetime.strptime(datestr, format)
            dt = dt.replace(tzinfo = sf_bike.USTimeZone())
            holiday_datetime_list.append(dt)

        self.holidays = holiday_datetime_list
        return self.holidays

    ####################################
    def read_sunrise(self):
        """ Read the times of sunrise and sunset for each day of the year"""
        filename = 'sunrise_2014.txt'
        table = ascii.read(filename)
        format = "%m/%d/%y %H:%M:%S"

        sunset_times = []
        sunrise_times = []
        for day in (table):
            # extract sunrise and sunset datetimes from the table.
            dt_sunrise = datetime.datetime.strptime(
                day['Date'] + ' ' + day['Sunrise(UTC)'], format)
            dt_sunrise = dt_sunrise.replace(tzinfo= sf_bike.UTC())

            dt_sunset = datetime.datetime.strptime(
                day['Date'] + ' ' + day['Sunset(UTC)'], format)
            dt_sunset = dt_sunset.replace(tzinfo=sf_bike.UTC())

            # Need to add a day for sunset because it occurs after 
            # midnight in UTC time.
            dt_sunset = dt_sunset + datetime.timedelta(1)

            # Convert from UTC to Pacific time zone
            dt_sunset = dt_sunset.astimezone(sf_bike.USTimeZone())
            dt_sunrise = dt_sunrise.astimezone(sf_bike.USTimeZone())
            sunrise_times.append(dt_sunrise)
            sunset_times.append(dt_sunset)

        self.sunrise_times = sunrise_times
        self.sunset_times = sunset_times

    ####################################
    def sunlight_index(self, datetime_obj):
        """Calculate a sunlight index relative to sunrise/sunset times
        """
        dt = datetime_obj

        # Find the day-of-the-year (ranging from 0 to 365).
        # We'll use this as an index to the sunrise and sunset lists.
        ind = int(dt.strftime("%j")) - 1
        
        t_since_sunrise = dt - self.sunrise_times[ind]
        t_since_sunset = dt - self.sunset_times[ind]

        t_till_sunset = -1 * t_since_sunset

        # Determine if we're closer to sunrise or sunset
        if abs(t_since_sunrise) <= abs(t_till_sunset):
            index = t_since_sunrise.total_seconds()
        else:
            index = t_till_sunset.total_seconds()

        return index
                
####################################
####################################
def bootstrap(data, n_samples=999, method = 1):
    """Make a bootstrap resampling of time series data
    
    data: array/matrix where each row is e.g. one day's time series,
      and each column is one minute from the day (so the array size would
      be (n_days, n_times_per_day)
    
    result: array of size (n_samples, n_times_per_day) that gives the
      bootstrapped results.  For a given time, we can use the n_samples=999
      samples to get the average and variance of the time series at that
      time
    """
    n_examples, n_times = numpy.shape(data)

    # Method 1: Sample entire days.  
    if method == 1:
        # Create an index array where each index refers to a different row.  
        # This is a 1-D array.
        ind = numpy.random.choice(n_examples, size = n_samples, replace=True)
        result = data[ind]

    # Method 2: Create a sampling separately for each time-of-day.
    if method == 2:
        result = numpy.zeros((n_samples, n_times), dtype=data.dtype)

        # Create a 2D array of indices with the same shape as the output
        ind = numpy.random.choice(n_examples, size = (n_samples, n_times), 
                                  replace=True)

        # Each column of indices corresponds to a column of the output array
        for icol in range(n_times):
            # Select the correct column of index values.
            myind = ind[:,icol]

            # Sample the data from the correct column with the given
            # index values
            result[:,icol] = data[myind, icol]

            pass

    return result
    

####################################
####################################
def check_bootstrap_iid(data):
    """Check whether data to be fed to bootstrap() are iid
    """
    acf_rand = stats.acf(numpy.random.randn(len(data[:,0])))
    sigma_acf_rand = numpy.std(acf_rand)
    for icol in range(len(data[0,:])):
        arr = data[:,icol]
#         corr = signal.fftconvolve(arr, arr[::-1], mode='full')
#         autocorr = corr[len(corr)/2:]
        acf = stats.acf(arr, fft=True, nlags=len(arr))
        if icol in [100]:
            pyplot.subplot(211)
            pyplot.plot(arr)
            pyplot.subplot(212)
            pyplot.plot(acf)
            pyplot.hlines([-sigma_acf_rand,sigma_acf_rand],0,30)
            pyplot.show()


####################################
####################################
def predict_simple(data, station_id, timestring, fmt='%Y-%m-%d %H:%M:%S',
                   quantity = 'bikes_available'):
    """Use simple extrapolation of weekly-folded bike data to estimate future 
    bike (or stand) availability"""
    
    # select adjacent times within this many minutes
    time_window_width = 5.0
    
    # Get just the station of interest
    dd = data[data['station_id'] == station_id]

    # Convert timestring to time of week
    TOW = time_of_week(timestring, fmt=fmt)
    
    # Select the data at the desired time.
    w_select = (dd["t_norm"] > TOW - time_window_width) * \
        (dd['t_norm'] < TOW + time_window_width)

    # Return a Prediction object
    return PredictionObj(dd[w_select][quantity], quantity)


####################################
####################################
# Example "time": "Tuesday 14:30"
def plot_dist_at_time(data, time, station_id = 68):
    """Plot the distribution of the number of bikes available at a given 
    time"""

    # Get just the station of interest
    data = data[data['station_id'] == station_id]

    # select all times within 5 minutes (time tolerance)
    t_tol = 2.0

    # Find the time-of-week corresponding to the given time
    TOW = time_of_week(time)

    w_select = (data["t_norm"] > TOW - t_tol) * \
        (data['t_norm'] < TOW + t_tol)

    ## print the 10th, 50th, and 90th percentiles of the distribution
    print numpy.percentile(data[w_select]["bikes_available"], [10,50,90])
    
    pyplot.hist(data[w_select]["bikes_available"])
    pyplot.show()


####################################
####################################
minutes_per_day = 1440.
def time_of_week(timestring, fmt='Day time'):
    """Convert a timestring to a time-of-week in minutes"""
    
    # example: "Monday 14:25"
    if fmt == 'Day time':
        # Using an arbitrary year, find a date with the given day
        for day_of_year in range(7):
            day_of_year += 1

            # add a zero-padded 3-digit day-of-year to the timestring
            ts = timestring + ' ' + day_of_year.__format__('03')
            # also add a year (for no good reason)
            ts = ts  +' 2014'
            dt = datetime.datetime.strptime(ts, '%A %H:%M %j %Y')

            # Does this datetime object's day-of-week match the one we 
            # requested?
            if strmatch(dt.strftime('%a')[0:2], timestring):
                # if we have a match, break the for loop and use this
                # datetime object
                break
        pass

    else:
        dt = datetime.datetime.strptime(timestring, fmt)
        pass

    # get the day of week
    dow = dt.strftime('%w')

    # Convert to time, in minutes, from start of the week
    t_norm = (float(dow) * minutes_per_day) + \
        dt.hour * 60.0 + float(dt.minute) + dt.second / 60.

    return t_norm

def time_of_day(timestring, fmt='Day time'):
    """Convert a timestring to a time-of-week in minutes

    NB: "Day Time" format not tested!!
    """
    
    # example: "Monday 14:25"
    if fmt == 'Day time':
        # Using an arbitrary year, find a date with the given day
        for day_of_year in range(7):
            day_of_year += 1

            # add a zero-padded 3-digit day-of-year to the timestring
            ts = timestring + ' ' + day_of_year.__format__('03')
            # also add a year (for no good reason)
            ts = ts  +' 2014'
            dt = datetime.datetime.strptime(ts, '%A %H:%M %j %Y')

            # Does this datetime object's day-of-week match the one we 
            # requested?
            if strmatch(dt.strftime('%a')[0:2], timestring):
                # if we have a match, break the for loop and use this
                # datetime object
                break
        pass

    else:
        dt = datetime.datetime.strptime(timestring, fmt)
        pass

    # get the time of day, in minutes
    # Convert to time, in minutes, from start of the week
    t_norm = \
        dt.hour * 60.0 + float(dt.minute) + dt.second / 60.

    return t_norm


####################################
####################################
def add_rows_to_table(table, values, insert_index = -1, keys=[]):
    """Add multiple rows to a table. The built-in 'add_row()' only adds
    one row at a time.
    
    values: A list of numpy arrays.  Each element of the list corresponds
    to one column of the Table.

    insert_index: Index of the row where new values will be inserted. The
    values are inserted just before this row. Default: at the end
    
    keys: Column names corresponding to the values (need not be in 
    the same order as table.colnames)
    
    """
    # By default, insert new data at the end of the table.
    if insert_index < 0:
        insert_index = len(table)

    colnames = table.colnames  # list of column names
    colvals = table.columns.values()  # list of arrays of column values
    
    # Assume that the new values are given in the correct order of the
    # table column keys
    if len(keys) == 0:
        keys = colnames
        if len(values) != len(keys):
            # throw exception
            raise Exception("Number of columns implied by `values` is wrong")

    # This will be the new container for the table's columns
    columns = table.TableColumns()

    # Loop over keys provided (which should correspond to columns)
    # and add values to each column
    for i, key in enumerate(keys):
        # Find the key within the list of column names
        ind = colnames.index(key)

        if key == 't_norm':
            if (values[i])[0] > MINUTES_PER_WEEK:
                print "MMMMMMMMMMMMMMMMMMMM", values[i]

        # Find the corresponding column value array, and insert our
        # new values to it
        mycol = colvals[ind]    # existing column
        newcol = mycol.insert(insert_index, values[i])
    
        Column.col_setattr(newcol, 'parent_table', table)
###        print values[ind], len(mycol), len(newcol)

        columns[key] = newcol
        

    table.columns = columns
    
####################################
####################################
def test(dat= None):
    ## Read data
    SFBD = SFBikeData(rebalance_data=dat)
    table = SFBD.rebalancing_data

    SFBD.clean_ts_all()

#     xx = SFBD.get_clean_ts(50)

#     yy = SFBD.get_clean_ts(68)

#     print 'LLLLLLL', numpy.min(yy.data), \
#         numpy.max(numpy.real_if_close(yy.data))

#     pyplot.subplot(211)
#     pyplot.plot(xx.data)
#     pyplot.subplot(212)
# #    pyplot.plot(SFBD.temp_timeseries)
#     pyplot.plot(yy.data)
#     pyplot.show()

##    print yy.euclidean_distance(xx)


    return SFBD

#     if dat==None:
#         dat = sf_bike.read_rebalance(unpickle=1)
#     print 'done reading data'

#     ## Test simple prediction
#     myID = 68
#     timestring = '2014-07-22 07:25:01'
#     PO = predict_simple(dat, myID, timestring)
#     print PO.percentile([10, 16, 50, 84, 90])
#     print PO.prob_atleast(4)

    pass

def test_simple2(timestring='2014-05-01 12:24:01', dat=None, plot=0):
    """Test predice_simple2() method"""
    if dat is None:
        SFBD = SFBikeData(unpickle=1)
    else:
        SFBD = dat
    print "doing it"
#     PO = SFBD.predict_simple2(50,timestring,
#                               tnorm=SFBD.common_timeline.tnorm)
    PO = SFBD.predict_simple2(50,timestring,
                              tnorm=SFBD.common_timeline.tnorm)
    print "done doing"
    print PO.percentile([10,16,50,84,90])
    print PO.prob_atleast(4)
    
    if plot > 0:
        PO.plot_dist()
    return SFBD



from scipy import signal
def test2():
    # test out convolution functions for autocorrelation

    x = numpy.arange(100.0) / (2*numpy.pi)
    y = numpy.sin(x)

    pyplot.plot(x,y)
    pyplot.show()

##    corr = numpy.correlate(y,y, mode='full')
    corr = signal.fftconvolve(y, y[::-1], mode='full')
    xx = numpy.arange(-len(y)+1, len(y)) / (2*numpy.pi)
    print len(corr), len(xx)
    pyplot.plot(xx, corr)
    pyplot.show()


def test_minute_alignment(method = 1):
    """Test whether the minutes are appropriately aligned for all stations
    """
    SFBD = SFBikeData(unpickle=1)
    alltimes = SFBD.rebalancing_data['time']
    timelist = []
    tz = sf_bike.USTimeZone()

    if method == 0:
        print "getting timelist..."
        for time in alltimes:
            timelist.append(datetime.datetime.
                            fromtimestamp(time.astype(int),tz) )
            
            print "Done getting timelist. Now convert to array..."
            timearr = numpy.array(timelist)
            
            print "...done.  Now checking times"
            for ID in numpy.unique(SFBD.rebalancing_data['station_id']):
                wid = SFBD.rebalancing_data['station_id'] == ID
                station_times = timearr[wid]
                for i in range(len(station_times)):
                    # compare this station to the very first station
                    dt = station_times[i] - timearr[i]
                    
                    if numpy.abs(dt.total_seconds()) > 60:
                        print "bad time...", i
                        
    if method == 1:
        w2 = SFBD.rebalancing_data['station_id'] == 2
        times2 = alltimes[w2]
        for ID in numpy.unique(SFBD.rebalancing_data['station_id']):
            print ID,
            sys.stdout.flush()
            wid = SFBD.rebalancing_data['station_id'] == ID
            station_times = alltimes[wid]
            maxdiff = 0
            diff = times2 - station_times
            print "MAXDIFF ", numpy.max(diff)
            
#             for i, time in enumerate(station_times):
#                 dt = alltimes[i] - station_times[i]
#                 if dt > 59:
#                     print "Bad Time"
#                 if dt > maxdiff:
#                     maxdiff = dt
#             print "MAX DIFFERENCE", maxdiff


def test_timeline(dat = None):
    """Tests to make sure the Timeline class is working"""
    SFBD = SFBikeData(unpickle = 1, rebalance_data=dat)
    table = SFBD.rebalancing_data

    # Create timeline object
    tl = Timeline(table['time'][0:50000])
    tl.make_datetimes()
    tl.find_all_daystarts()
    print tl.ind_daystart
    print tl.datetimes[tl.ind_daystart[1]]

    # test periodic normalization of the timeline
    per = tl.tnorm_periodic(1440)
##    return tl
    

    # Test whether find_daystart() method is working
    print "find_daystart():"
    input_time = tl.times[6200]
    ind = tl.find_daystart(input_time)
    print tl.datetimes[ind]

    # Test whether find_dayend() method is working
    print "find_dayend():"
    ind = tl.find_dayend(input_time)
    print tl.datetimes[ind]
    ind = tl.find_dayend(ind_start = 6200)
    print tl.datetimes[ind]

    # Test ability to find days of the week
    days_of_week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    ind, indend = tl.find_day_of_week(5)  # Saturdays = 5
    for ii in range(len(ind)):
        dt = tl.datetimes[ind[ii]]
        print "Day of week:", dt.weekday(), dt.strftime("%c")

    print ""
    ind, indend = tl.find_weekdays()
    for ii in range(len(ind)):
        dt = tl.datetimes[ind[ii]]
        print "Day of week:", dt.weekday(), dt.strftime("%c")
        dt = tl.datetimes[indend[ii]]
        print "            ", dt.weekday(), dt.strftime("%c")

    return tl


def test_neighbors(SFBD=None):
    """Test neighbor-finding, friends-of-friends"""

    if SFBD is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
            station.normalize_ts()
        
        # Find neighbors for all stations
        print "Finding neighbors for all stations"
        for station in all_stations:
            station.find_neighbors(all_stations)
        
    # The following line is here solely because we want to be able to use
    # SFBD as an input when debugging.
    all_stations = SFBD.all_stations.station_list

    # Compile a big array of all distances 
    all_dist = numpy.array([])
    for station in all_stations:
        # Create a big array of all distances
        all_dist = numpy.concatenate((all_dist,station.neighbor_dist))

    # Get rid of bad and large distances, then plot them
    all_dist = all_dist[all_dist < 0.002]
#     pyplot.hist(all_dist, 50)
#     pyplot.show(block=False)
    print "Characteristic distances:", \
        numpy.mean(all_dist), numpy.median(all_dist), numpy.std(all_dist)

    # Run FoF 
    print "Running Fof 0.00024"
    SFBD.friends_of_friends(0.00024)
    print "Running Fof 0.00035"
    SFBD.friends_of_friends(0.00035)
#     print "Running Fof 0.00045"
#     SFBD.friends_of_friends(0.00045)

    for station in all_stations:
        print station.ID, station.cluster_id

    return SFBD


def test_kmeans(SFBD=None):
    # Test k-means clustering algorithm
    if SFBD is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
            station.normalize_ts()
        
#         # Find neighbors for all stations
#         print "Finding neighbors for all stations"
#         for station in all_stations:
#             station.find_neighbors(all_stations)
        
    # The following line is here solely because we want to be able to use
    # SFBD as an input when debugging.
    all_stations = SFBD.all_stations.station_list
  
    print "Running K-means..."
    centroids, idx = SFBD.kmeans(32)
    return centroids, idx, SFBD
    

def test_station_objects():
    SFBD = SFBikeData(unpickle=1)
    all_stations = SFBD.all_stations
    station = all_stations.get_station(50)

    station.normalize_ts()
    print "Normalized."

    station.clean_ts()
    print "cleaned"

    station.create_days()
    print 'days_created'
    
#     # Change station data to see if it will change the original table
#     # data as well
#     station.table['bikes_available'][0] = -1
#     print station.table
#     print SFBD.rebalancing_data
#     return SFBD, station

    day = station.daylist[10]

    other_day_list = station.daylist
    day.find_neighbors(other_day_list)
    print 'day neighbors found'
##    print day.neighbor_dist
    
    print "Plotting day timeseries"
    ind = 3
    # Show the 'cleaned' timeseries, which includes fourier filtering
    pyplot.plot(day.neighbor_list[ind].table['ts_clean'], 'b--')
    pyplot.plot(day.table['ts_clean'], 'g--')
##    pyplot.show()

    # also plot the normalized timeseries without filtereing
    pyplot.plot(day.neighbor_list[ind].table['ts_norm'], 'b-')
    pyplot.plot(day.table['ts_norm'], 'g-')
    pyplot.show()


#    day.plot_ts()

#     print "Day table"
#     print day.table

    
    return station,day


def test_season_fix( station=None):
    """Tests of correction for seasonal trends"""

    if station is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations
        station = all_stations.get_station(50)

        station.normalize_ts()
        print "Normalized."

        station.clean_ts()
        print "cleaned"

        station.create_days()
        print 'days_created'

    # Create time series from the station's data
    ts = TimeSeries(station.ts_norm)

    # get the autocorrelation function and plot it
    corr = signal.fftconvolve(ts.data, ts.data[::-1], mode='full')
    autocorr = corr[len(corr)/2:]
    xdat = numpy.arange(len(autocorr)) / 60.0
    pyplot.plot(xdat, autocorr)
    pyplot.show()

    # Define the period length of 1 day
    period = station.timeline.ind_daystart[1] - \
        station.timeline.ind_daystart[0]

    # Find the start of the first full day
    start_index = station.timeline.ind_daystart[0]

    # 1-day correction
    period = station.timeline.ind_daystart[1] - \
        station.timeline.ind_daystart[0]

    folded = ts.fold(period, start_index)

    arr = ts.seasonal_correct(period, start_index, plot=1)
    ts = TimeSeries(arr)
    pyplot.plot(arr)
    pyplot.show()

    return folded
    
    
    
################################
def test_bootstrap(SFBD=None):
    """Test out bootstrapping routines"""
    if SFBD is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
            station.normalize_ts()
            station.create_days()

        

    # The following line is here solely because we want to be able to use
    # SFBD as an input when debugging.
    all_stations = SFBD.all_stations.station_list
  
    print "Running bootstrap"
#     for stat in all_stations:
#         stat.bootstrap_day_of_week()


    stat = SFBD.all_stations.get_station(50)

    ################
    boot, dat = stat.bootstrap_day_of_week()

##    print "ndocks:", stat.basic_data['dockcount']
    
    # plot bootstrapped results
    iday = 0   # 0 == Monday

##    stat.plot_bootstrap_day(iday)

##    return stat

    #################
    # Regression
    stat.extra_features = None
    stat.regress([0.1], [100.0], 10000, 
                 selection_criterion='crossval')
    stat.plot_prediction_boot()

    stat.quicksave("station50.pkl")
    return stat

#     # correct bad values
#     wbad = boot[iday] < 0.0
#     (boot[iday])[wbad] = numpy.nan

#     pyplot.subplot(211)
#     stat.plot_bootstrap_as_2dhist(boot[iday])
        
#     # get percentiles to overplot
#     my_percentiles = [10, 50, 90]
#     perc = numpy.nanpercentile(boot[iday], my_percentiles, axis=0)
#     for pp in perc:
#         pyplot.plot(pp, 'blue', linewidth=2.5)

##    pyplot.show()

    boot_all = boot

#     print "RETURNING EARLY"
#     return boot, dat

#     ################## Now look at all weekdays combined bootstrap
#     boot = stat.bootstrap_weekdays()
#     wbad = boot < 0.0
#     boot[wbad] = numpy.nan

#     pyplot.subplot(212)
# #     for i, daydat in enumerate(boot):
# #         # only plot a fraction of the results
# #         if i % 10000 == 0:
# #             pyplot.plot(daydat, 'gray', linewidth=0.2)

#     stat.plot_bootstrap_as_2dhist(boot)

#     # get percentiles to plot
#     nfilter = 100
#     my_percentiles = [10, 50, 90]
#     perc = numpy.nanpercentile(boot, my_percentiles, axis=0)
#     for pp in perc:
#         pyplot.plot(filters.median_filter(pp,nfilter), 
#                     'blue', linewidth=2.5)

#     pyplot.show()

    
#     ####### 
#     # Now make a plot for each weekday, with a comparison of the 
#     # percentiles from the combined weekday bootstrap
#     for iday in range(5):
#         wbad = boot_all[iday] < 0.0
#         (boot_all[iday])[wbad] = numpy.nan
#         pyplot.subplot(511 + iday)

#         stat.plot_bootstrap_as_2dhist(boot_all[iday])

#         perc1 = numpy.nanpercentile(boot_all[iday], my_percentiles,axis=0)
#         for pp in perc1:
#             pyplot.plot(filters.median_filter(pp,nfilter),
#                         'blue',linewidth=2.5)

#         # Overplot the combined weekday bootstrap percentiles
#         for pp in perc:
#             pyplot.plot(filters.median_filter(pp,nfilter),
#                         'yellow', linestyle='dashed', linewidth = 1.5)
#     pyplot.show()
#     return boot,dat


######################
######################
######################
def test_regress(SFBD=None):
    """Test out bootstrapping routines"""
    if SFBD is None:
        SFBD = SFBikeData(unpickle=1, timeline_filename='common_timeline.pkl')
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
#            station.normalize_ts()
#            station.create_days()
            pass
        
#     print "making features"
    extra_feat = ['week_of_year','holiday','sunrise','weather']
#     SFBD.common_timeline.make_features(
#         extra_features=extra_feat)
#     print 'done', numpy.shape(SFBD.common_timeline.get_features())


    # The following line is here solely because we want to be able to use
    # SFBD as an input when debugging.
    all_stations = SFBD.all_stations.station_list
  
    stat = SFBD.all_stations.get_station(50)
    stat.normalize_ts()

    ff = stat.get_features()
    stat.extra_features = extra_feat
    ff2 = stat.get_features()
    print 'ff', numpy.shape(ff), numpy.shape(ff2)

    print "Running regression"

##    SFBD.regress_all_stations()

#     stat.regress([0.001,0.01,0.1], [10.,100, 1000.], 2000,  
#                  selection_criterion='crossval')

#     stat.extra_features = None
#     stat.regress([0.001, 1.0, 10.0], [0.1, 1.0, 10., 1000.], 5000, 
#                  selection_criterion='crossval')


    stat.extra_features = ['holiday','sunrise','weather']
    stat.regress([.1], [10.0], 5000,
                 selection_criterion='crossval')

##    stat.learning_curves()

##    stat.test_regress()

    print "done regress"


    # Plot predicted values on top of the real deal
    stat.plot_prediction_comparison()

    return stat

    

def train_svr_all(SFBD=None):

    # Load up the SF bike data
    if SFBD is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
            station.normalize_ts()
            station.create_days()
    
    print "Running lots of regressions"
    SFBD.regress_all_stations()

    print "Saving data"
    SFBD.save_data(do_timeline=True, do_stationlist=True)
    
    return SFBD



def test_station(stationID=50, SFBD=None):
    """Read the data and return one station
    """
    timeline = Timeline(0, load_filename='common_timeline.pkl')
    file = open('station50.pkl','rb')
    stat = Station(0,0,0,timeline, loadfile=file)
    file.close()

    stat.create_days()

    stat.bootstrap_day_of_week()

#    stat.plot_bootstrap_day(0)

#    stat.plot_day_stack(1)

    ax = stat.plot_bootstrap_week()

##    stat.plot_double_boot(2)

##    ax = stat.plot_bootstrap_day(1, do_percentiles=True, do_show=True)
    return stat



def save_processed_data(SFBD=None):
    if SFBD is None:
        SFBD = SFBikeData(unpickle=1)
        all_stations = SFBD.all_stations.station_list
        # Normalize all stations
        print "Normalizing stations"
        for station in all_stations:
            station.normalize_ts()
            station.create_days()

#     print "Running lots of regressions"
#     SFBD.regress_all_stations()

    stat = SFBD.all_stations.get_station(50)
    stat.extra_features = None
    stat.regress([0.1], [100.0], 10000,
                 selection_criterion='crossval')

    print "Saving data"
    stat.quicksave('station50.pkl')
    SFBD.save_data(do_timeline=True, do_stationlist=True)
    
    
    pass


def demo():
    """Demonstration for Insight interview
    """
    
    # Load data structures
    SFBD = SFBikeData(unpickle=1, timeline_filename='common_timeline.pkl',
                      stationlist_filename = 'stationlist.pkl')

    station20 = SFBD.all_stations.get_station(21)
    station20.create_days()

    file = open('station50.pkl','rb')
    station50 = Station(0,0,0, timeline=SFBD.common_timeline,
                        loadfile = file)
    file.close()
    station50.create_days()

    # Do the bootstrapping
    print "Running bootstrap"
    station50.bootstrap_day_of_week()
    station20.bootstrap_day_of_week()

    # Show time series for one station
    print "Showing example time series"
    station50.plot_ts()
    raw_input("press enter:")

    # Plot a bunch of e.g. Tuesdays to demonstrate bootstrapping
    station50.plot_day_stack(1)
    raw_input("press enter:")

    # Show 1-day results of bootstrapping
    print "Showing 1 day bootstrap for station 50"
    station50.plot_bootstrap_day(1)
    raw_input("press enter:")

    # Show 1-week results of bootstrapping
    print "Showing 1 week bootstrap for station 50"
    station50.plot_bootstrap_week()
    raw_input("press enter:")

    # Show 1-week results for another station
    print "Showing 1 week bootstrap for station 21"
    station20.plot_bootstrap_week()
    raw_input("press enter:")

    # Show comparison w/ SVM.
    print "Showing SVM results"
    station50.plot_prediction_boot()
    raw_input("press enter:")
    
    # Do a prediction
    "Doing prediction"
    pred_obj = station50.predict('2015-09-16 15:35:22')
    print "10th, 50th, 90th percentiles:", pred_obj.percentile([10, 50, 90])
    pred_obj.plot_dist()

    return station50

    pass
