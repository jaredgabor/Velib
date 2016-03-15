import numpy
import urllib2 as URL
import json
import cPickle as pickle
import os

##import sys
##print sys.path

# Import the PyTables module for dealing w/ HDF5 data structures
import tables
import time

class Global:
    def __init__(self):
        self.datafile = "velib_database.h5"
        self.data_count_file = "n_data.txt"
        self.n_stations = 1230 + 100 ## 100 extra as a buffer
        self.nchar_address = 128
        self.base_time = 1429194331  # April 16 2015

glob = Global()

def fetch_velib_auto():
    """Read velib data, append it to an existing pickle file"""
    # This try statement guards against the lack of internet connection
    try:
        dat = get_velib_data()
    except URL.URLError as err:
        print "URLError: No internet connection?"
        return 0

    save_velib_data(dat, glob.datafile)
##    update_data_count()

def get_velib_data():
    """Read velib data one time"""
    api_url = "https://api.jcdecaux.com/vls/v1/"
    query_string = "stations?contract=Paris&apiKey="
    api_key = "ec29d3b17e5162e1459aaad45cddfe74fe832379"
    my_url = api_url + query_string + api_key

    urlobj = URL.urlopen(my_url)
    data = json.load(urlobj)
#    data = urlobj.read()
#    help(data)
    return data

def separate_static_dynamic(data):
    """Separate the JSON-format data read from Velib into 2 objects, one
    for static data and one for dynamic data"""
    dynamic_data = DDat(data)
    static_data = SDat(data)

    return static_data, dynamic_data

def save_velib_data(data, outfile_name):
    """Save velib data in a pickle file."""
    # separate static and dynamic information in the input data
    d_static, d_dyn = separate_static_dynamic(data)

    # Is there already a data file?  If so, then add the current data to it
    if os.path.isfile(outfile_name):

        # Extract existing static information, and compare to make sure things
        # haven't changed.
        outfile = tables.open_file(outfile_name, "a")
        stat_old = outfile.root.VelibData.static
        indices = d_static.match_static_data(stat_old)
        d_dyn.match_dynamic_data(indices)

        # Add dynamic data to the table        
        table = outfile.root.VelibData.dynamic
        d_dyn.populate_dyn(table)

    else :
        # There is no existing data file, so create a new one
        outfile = tables.open_file(outfile_name, "w")

        # create a "group" to hold the data
        group = outfile.create_group("/", "VelibData", "All Velib data")
        
        # 2 tables -- 1 for static data and 1 for dynamic data
        table_stat = outfile.create_table(group, "static", 
                                          StaticDat, "Static data")
        table_dyn = outfile.create_table(group, "dynamic", 
                                         DynamicDat, "Dynamic data")

        # populate the tables with data
        d_static.populate_stat(table_stat)
        d_dyn.populate_dyn(table_dyn)

    outfile.close()


def read_database():
    """Read the database file on disk and print out test info """
    file = tables.open_file(glob.datafile)
    table_d = file.root.VelibData.dynamic
    table_s = file.root.VelibData.static
    n_rows = len(table_d)
    print "Nrows in dynamic table:", n_rows
    print "N stations:", len(table_d[0]["last_update"])
    print "Time of most recent sampling:", \
        time.asctime(time.localtime(recover_time(table_d[-1]["sample_time"])))
    print "Nbikes available at most recent sampling:", \
        table_d[n_rows-1]["available_bikes"]
    print "Time of last_update at most recent sampling:", \
        time.asctime(
        time.localtime(recover_time(table_d[n_rows-1]["last_update"][0])))
    print "Number arr", table_s[0]["number"]
    file.close()

class SDat():
    """Class to hold static Velib data"""
    def __init__(self, data):
        n_stations = glob.n_stations
##        n_stations = len(data)
##        assert n_stations == glob.n_stations

        # create arrays
        self.address = numpy.zeros(n_stations, 
                                   dtype=(numpy.unicode, glob.nchar_address))
        self.bike_stands = numpy.zeros(n_stations, dtype=numpy.int16)
        self.number = numpy.zeros(n_stations, dtype=numpy.int)
        self.position = numpy.zeros((n_stations,2), dtype=numpy.int64)

        # populate the arrays with real data
        i=0
        for station in data:
            self.address[i] = station["address"]
            self.bike_stands[i] = station["bike_stands"]
            self.number[i] = station["number"]
            self.position[i,0] = (station["position"])["lat"]
            self.position[i,1] = (station["position"])["lng"]
            i += 1

    def populate_stat(self, table):
        """Populate a static data Tables table with data"""
        myrow = table.row
        # HDF5 doesn't handle unicode strings, so we need to convert to 
        # *byte* strings, which we can put in the HDF5 file 
        addy = numpy.zeros(len(self.address), 
                           dtype=(numpy.str, glob.nchar_address))
        for i in range(len(addy)):
            addy[i] = (self.address[i]).encode('utf8')

        myrow["address"] = addy
        myrow["bike_stands"] = self.bike_stands
        myrow["number"] = self.number
        myrow["position"] = self.position
        myrow.append()
        table.flush()
    
    def match_static_data(self, static_table):
        """Check whether the current static data match the old ones"""
        stationIDs = static_table[0]["number"]
#         if not numpy.array_equal(stationIDs, self.number):
#             print "ERROR in new data! Bad station ID numbers"
        indices = []
        for i in range(len(stationIDs)):
            if(stationIDs[i] == 0): continue
            w = numpy.where(self.number == stationIDs[i])[0]
            nmatch = len(w)
            if nmatch > 1:
                print "TOO MANY MATCHES?", nmatch, stationIDs[i]
                jmatch = w[0]
            if nmatch == 1:
                jmatch = w[0]
            else:
                jmatch = -1
            indices.append(jmatch)

        # Add blanks to fill out the array to length "glob.n_stations".
        # Blank entries will be assigned index -1
        indices = numpy.array(indices)
        print "TESTING INDIC 0", len(indices), glob.n_stations - len(indices)
        indices = numpy.append(indices, 
                               numpy.zeros(glob.n_stations-len(indices),
                                           dtype='int') - 1)


        return indices

class DDat():
    """Class to hold dynamic Velib data from a single API data request 
    in a compact form"""
    def __init__(self, data):
        n_stations = glob.n_stations
##        n_stations = len(data)      
##        assert n_stations == glob.n_stations

        # create arrays
        self.available_bike_stands = numpy.zeros(n_stations, numpy.int16)
        self.available_bikes = numpy.zeros(n_stations, numpy.int16)
        self.last_update = numpy.zeros(n_stations, numpy.int32)
        self.status = numpy.zeros(n_stations, numpy.bool)

        # Set last elements in arrays to null values
        self.available_bike_stands[-1] = -1
        self.available_bikes[-1] = -1
        self.last_update[-1] = -1
        self.status[-1] = False

        # populate arrays with data
        i = 0
        for station in data:
            self.available_bike_stands[i] = station["available_bike_stands"]
            self.available_bikes[i] = station["available_bikes"]
            self.last_update[i] = reduce_time_bytes(station["last_update"])
            if station["status"] == "OPEN":
                self.status[i] = True
            else:
                self.status[i] = False
            i += 1

    def populate_dyn(self, table):
        """Populate the dynamic Tables table with real data"""
        myrow = table.row
        myrow["sample_time"] = int(time.time() - glob.base_time)
        myrow["available_bike_stands"] = self.available_bike_stands
        myrow["available_bikes"] = self.available_bikes
        myrow["last_update"] = self.last_update
        myrow["status"] = self.status
        myrow.append()
        table.flush()

    def match_dynamic_data(self, indices):
        """Match the current dynamic data to the old data using indices"""
        self.available_bike_stands = self.available_bike_stands[indices]
        self.availabel_bikes = self.available_bikes[indices]
        self.last_update = self.last_update[indices]
        self.status = self.status[indices]


def reduce_time_bytes(time_in_ms):
    """Reduce the number of bytes required to store time data"""
    time_in_s = time_in_ms / 1000
    time_adjust = time_in_s - glob.base_time
    return time_adjust


def recover_time(adjusted_time):
    """Find the time from a time integer value that has been adjusted to reduce
    the memory size."""
    time_in_s = adjusted_time + glob.base_time
    return time_in_s
    

myshape = (glob.n_stations,)
class StaticDat(tables.IsDescription):
    """Tables module descriptor for static data table"""
    # "Shape" of each array
    address = tables.StringCol(glob.nchar_address, shape = myshape)
    bike_stands = tables.Int32Col(shape=myshape)
    number = tables.Int32Col(shape=myshape)
    position = tables.Float64Col(shape = (glob.n_stations, 2))
    

class DynamicDat(tables.IsDescription):
    """Tables module descriptor for dynamic data table"""
    sample_time = tables.Int32Col()
    available_bike_stands = tables.Int32Col(shape=myshape)
    available_bikes = tables.Int32Col(shape=myshape)
    last_update = tables.Int32Col(shape=myshape)
    status = tables.BoolCol(shape=myshape)


# def update_data_count():
#     """Open pickle file with 1 integer, increment the int, then re-save it"""
#     counter_file = glob.data_count_file
#     if os.path.isfile(counter_file):
#         file = open(counter_file,"r+")
#         # need to check if the file is empty first?
#         ndat = int(file.readline())
#     else:
#         ndat = 0
#         file = open(counter_file, "w")

#     # increment counter
#     ndat += 1

#     # save output
#     file.seek(0)
#     file.write(str(ndat))
#     file.close()
#     print "No. of data entries:", ndat
    

# def read_velib_data_file(filename):
#     """Read an entire velib data file in pickle format"""

#     # First read the number of data entries
#     counter_file = glob.data_count_file
#     cfile = open(counter_file, "r")
#     n_data = int(cfile.readline())
#     cfile.close()

#     # open the actual data file and read all the data
#     data = []
#     file = open(filename, 'rb')
#     for i in range(n_data):
#         d = pickle.load(file)
#         data.append(d)

#     file.close()
#     return data



# Make this module callable
if __name__ == "__main__":
##    import os
    os.chdir("/Users/jgabor/Documents/Junk/VELIB/DATA")

    ## check file size
    fsize = os.path.getsize(glob.datafile)
    if float(fsize) < 1e9:
        ## fetch the data
        fetch_velib_auto()
    else:
        pass

    os.chdir("/Users/jgabor/")
