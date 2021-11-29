import pandas as pd
import numpy as np
from dateutil.parser import parse as parse_date
import datetime
import matplotlib.pyplot as plt
from error_unpacker import ErrorUnpacker

parse_dates = False

# import csv data to pandas dataframe
df = pd.read_csv('snuffelfiets_data.csv')

# get relevant columns from dataframe
particulate_10 = np.array(list(df['pm10']))  # ug/M^3
particulate_2_5 = np.array(list(df['pm2_5']))  # ug/M^3
particulate_1_0 = np.array(list(df['pm1_0']))  # ug/M^3
nitrous_oxide = np.array(list(df['no2']))  # ug/M^3
volatile_organic_compounds = np.array(list(df['voc']))  # TODO: find out unit for VOC measurement

airpressure = np.array(list(df['pressure']))  # hPa
temperature = np.array(list(df['temperature']))  # degrees Celsius
humidity = np.array(list(df['humidity']))  # percentage

lat = np.array(list(df['lat']))
lon = np.array(list(df['lon']))

errors = np.array(list(df['error_code']))

if parse_dates:
    time = np.array([datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
                     for time_string in list(df['recording_time'])])
else:
    time = np.array(list(df['recording_time']))

# get valid gps entries (nonzero coordinates)
valid_lat_indices = np.where(lat != 0)
valid_lon_indices = np.where(lon != 0)

# get valid particulate sensor entries (no error)
valid_sensor_indices = ErrorUnpacker.filter_out_error_codes(errors, [16, 32, 128, 512, 1024, 2048, 4096])

# gps coordinates for map (assigned region)
min_lon = 5.7228
max_lon = 5.9680
min_lat = 50.9251
max_lat = 51.0796
map_bounds = (min_lon, max_lon, min_lat, max_lat)

# get gps values within assigned region
lat_indices_in_region = np.where(np.logical_and(min_lat < lat, lat < max_lat))
lon_indices_in_region = np.where(np.logical_and(min_lon < lon, lon < max_lon))

# get intersections of different filters
gps_indices_in_region = np.intersect1d(lat_indices_in_region, lon_indices_in_region)
valid_coordinates = np.intersect1d(valid_lat_indices, valid_lon_indices)
filter_results = np.intersect1d(valid_sensor_indices, np.intersect1d(valid_coordinates, gps_indices_in_region))

# output amount of entries matching filters
print('valid lat values:', len(valid_lat_indices[0]))
print('valid lon values:', len(valid_lon_indices[0]))
print('valid coordinates:', len(valid_coordinates))
print('valid sensor values:', len(valid_sensor_indices[0]))
print('gps coordinates in region:', len(gps_indices_in_region))
print('total filter results:', len(filter_results))

print('-')

ErrorUnpacker.print_error_report(errors)

# # set up dataframe for filtered data
# filtered_data = {'time': time[filter_results],
#                  'lat': lat[filter_results],
#                  'lon': lon[filter_results],
#                  }
# filtered_data = pd.DataFrame(filtered_data)
#
# # read map image
# map_image = plt.imread('map_sittard_geleen.png')
#
# # create plot window
# fig, ax = plt.subplots(figsize=(8, 7))
#
# # set axis limits
# ax.set_xlim(map_bounds[0], map_bounds[1])
# ax.set_ylim(map_bounds[2], map_bounds[3])
#
# # set graph title
# ax.set_title('Figure Title.')
#
# # set axis titles
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
#
# # enable legend
# ax.legend()
#
# # plot image
# ax.imshow(map_image, zorder=0, extent=map_bounds, aspect='equal')
#
# # show plot window
# plt.show()
