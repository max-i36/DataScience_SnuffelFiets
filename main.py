import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import csv data to pandas dataframe
df = pd.read_csv('snuffelfiets_data.csv')

# get relevant columns from dataframe
acc = np.array(list(df['acc_max']))
lat = np.array(list(df['lat']))
lon = np.array(list(df['lon']))
time = np.array(list(df['recording_time']))

# get valid value entries (nonzero)
valid_acc_indices = np.where(acc != 0)
valid_lat_indices = np.where(lat != 0)
valid_lon_indices = np.where(lon != 0)

# gps coordinates for map (assigned region)
minlon = 5.7228
maxlon = 5.9680
minlat = 50.9251
maxlat = 51.0796

map_bounds = (minlon, maxlon, minlat, maxlat)

# get gps values within assigned region
lat_indices_in_region = np.where(np.logical_and(minlat < lat, lat < maxlat))
lon_indices_in_region = np.where(np.logical_and(minlon < lon, lon < maxlon))

# get intersections of different filters
gps_indices_in_region = np.intersect1d(lat_indices_in_region, lon_indices_in_region)
valid_values = np.intersect1d(valid_acc_indices, np.intersect1d(valid_lat_indices, valid_lon_indices))
valid_values_in_region = np.intersect1d(gps_indices_in_region, valid_acc_indices)

# output amount of entries matching filters
print('valid acc values:', len(valid_acc_indices[0]))
print('valid lat values:', len(valid_lat_indices[0]))
print('valid lon values:', len(valid_lon_indices[0]))
print('valid value rows:', len(valid_values))
print('gps coordinates in region:', len(gps_indices_in_region))
print('valid values in region:', len(valid_values_in_region))

# set up dataframe for filtered data
filtered_data = {'acc': acc[valid_values_in_region],
                 'time': time[valid_values_in_region],
                 'lat': lat[valid_values_in_region],
                 'lon': lon[valid_values_in_region],
                 }
filtered_data = pd.DataFrame(filtered_data)

# divide entries by acc value
acc = np.array(list(filtered_data.acc))
lower_limit = 3
upper_limit = 7
low_indices = np.where(acc <= lower_limit)
mid_indices = np.where(np.logical_and(acc > lower_limit, acc <= upper_limit))
high_indices = np.where(acc > upper_limit)
lon = np.array(list(filtered_data.lon))
lat = np.array(list(filtered_data.lat))

# read map image
map_image = plt.imread('map_sittard_geleen.png')

# create plot window
fig, ax = plt.subplots(figsize=(8, 7))

# plot trackpoints with acc data
# ax.scatter(filtered_data.lon, filtered_data.lat, zorder=1, alpha=0.2, c='b', s=10)
ax.scatter(lon[low_indices], lat[low_indices], zorder=1, alpha=0.5, c='g', s=10, label='acc <= 3')
ax.scatter(lon[mid_indices], lat[mid_indices], zorder=1, alpha=0.5, c='y', s=10, label='3 < acc <= 7')
ax.scatter(lon[high_indices], lat[high_indices], zorder=1, alpha=0.5, c='r', s=10, label='acc > 7')

# set axis limits
ax.set_xlim(map_bounds[0], map_bounds[1])
ax.set_ylim(map_bounds[2], map_bounds[3])

# enable legend
ax.legend()

# plot image
ax.imshow(map_image, zorder=0, extent=map_bounds, aspect='equal')

# show plot window
plt.show()
