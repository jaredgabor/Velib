#
# Code for reading in weather data associated with bikes
#
import numpy
import matplotlib.pyplot as pyplot
from astropy.io import ascii
import datetime



class Weather:
    
    def __init__(self, file='201408_weather_data.csv', 
                 timezone=None):
        self.datafile = file
        self.timezone = timezone

        self.read_weather()
        self.make_datetimes()

        self.weather_features = None
        pass



    def read_weather(self):
        """Read ascii file to get weather info
        """
        print "Reading weather data from file",self.datafile
        tab = ascii.read(self.datafile)
        
        # Fix 'T' values in precipitation column, which represent tiny
        # amounts of rain (not measurable)
        TINY_VALUE = '.005'   # 0.005 is half the smallest measurable value
        rain = tab['PrecipitationIn']
        wbad =  (rain == 'T')
        rain[wbad] = TINY_VALUE
        rain = numpy.array(rain).astype("float")

        # Replace string version of precip with float version
        tab['PrecipIn'] = rain
        tab.remove_column('PrecipitationIn')

        self.table = tab


    def make_datetimes(self):
        # make datetimes out of the dates (which are given as strings)
        dates = []
        fmt = '%m/%d/%Y'
        for datestring in self.table['PDT']:
            dates.append(datetime.datetime.strptime(datestring,fmt))

        # Add timezone information to datetimes, if it exists.
        if self.timezone is not None:
            dates_tz = []
            for dt in dates:
                dates_tz.append(dt.replace(tzinfo=self.timezone))

        self.datetimes = dates


    ####################################
    def get_weather_features(self):
        """ Get features (for regression) based on this bikedata's weather data
        """
        if self.weather_features is None:
            raise Exception("Weather features not made yet.")
###            self.make_weather_features()
        else:
            return self.weather_features

    ####################################
    def make_weather_features(self, timeline_dt_list):
        """ Get features (for regression) based on the weather data
        
        timeline_dt_list: a list of datetimes from a Timeline object
        """

        print "Making weather features..."

        N_FEATURES = 2
        n_examples = len(timeline_dt_list)
        XX = numpy.zeros((n_examples, N_FEATURES))
        indices = numpy.zeros(n_examples,dtype='int')
        ind_weatherday = 0

        # Loop over all times in the timeline
        for ii, time in enumerate(timeline_dt_list):
            # Find where this time in the timeline matches the date
            # of some weather data.
            jj = ind_weatherday
            while time.date() != self.datetimes[jj].date():
                # Make sure jj does not get too large to be an index to
                # the list.
                # Note this is probably a bad idea to do it this way.
                if jj == len(self.datetimes)-1:
                    break
                jj += 1
##                print jj

            ind_weatherday = jj
            indices[ii] = ind_weatherday

#            XX[ii, 0] = self.table['PrecipIn'][ind_weatherday]
#            XX[ii, 1] = self.table['Mean TemperatureF'][ind_weatherday]
##            XX[ii, 2] = self.table['MeanDew PointF'][ind_weatherday]

        XX[:,0] = self.table['PrecipIn'][indices]
        XX[:,1] = self.table['Mean TemperatureF'][indices]
        self.weather_features = XX
        return XX

