import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from error_unpacker import ErrorUnpacker
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
import math

# parse_dates = False
parse_dates = True

# import snuffelfiets csv data to pandas dataframe
df = pd.read_csv('snuffelfiets_data.csv')
df.drop(['air_quality_observed_id',
         'geom',
         'recording_time',
         'voc',
         'acc_max',
         'no2',
         'horizontal_accuracy',
         'vertical_accuracy',
         ],
        axis=1,)

# import weather data to dataframe
df_weather = pd.read_csv('weerdata.csv')
# only keep relevant columns to save memory
df_weather = df_weather[['YYYYMMDD', 'DDVEC', 'FHVEC', '   PG']]
# rename columns to something more human-readable
df_weather.rename(columns={'YYYYMMDD': 'timestamp', 'DDVEC': 'wind_direction', 'FHVEC': 'wind_speed', '   PG': 'air_pressure'}, inplace=True)


# function to convert YYYYMMDD formatted integers to datetime objects
def int_to_timestamp(integer):
    string = str(integer)
    return datetime.datetime.strptime(string, '%Y%m%d')


# function to decompose wind information from meteorological degrees to vector components
def vector_wind(wind_direction, wind_speed):
    # convert direction to radians
    wind_direction = wind_direction/180*math.pi
    # get x and y components
    wind_x = pd.DataFrame(-wind_speed.abs()*wind_direction.apply(math.sin), columns=['wind_x'])
    wind_y = pd.DataFrame(-wind_speed.abs()*wind_direction.apply(math.cos), columns=['wind_y'])
    return pd.concat([wind_x, wind_y], axis=1)


# get wind vector components
wind_vectors = vector_wind(df_weather.wind_direction, df_weather.wind_speed)

# add wind vector components to weather dataframe
df_weather = pd.concat([df_weather, wind_vectors], axis=1)

# parse timestamp column using custom function
df_weather.timestamp = df_weather.timestamp.apply(int_to_timestamp)

# set display interval width
interval_width = 7
interval_duration = datetime.timedelta(days=interval_width)

# get valid particulate sensor entries (no error)
invalid_sensor_indices = ErrorUnpacker.filter_out_error_codes(df.error_code, [16, 32, 128, 512, 1024, 2048, 4096])

# print error report
print('Report of sensor error codes from Sniffer Bike data:')
ErrorUnpacker.print_error_report(df.error_code)

# drop records with sensor errors
df.drop(invalid_sensor_indices[0], axis=0, inplace=True)

# gps coordinates for map (assigned region)
min_lon = 5.7228
max_lon = 5.9680
min_lat = 50.9251
max_lat = 51.0796
map_bounds = (min_lon, max_lon, min_lat, max_lat)

# get gps values within assigned region
df.where(np.logical_and(min_lat < df.lat, df.lat < max_lat), inplace=True)
df.where(np.logical_and(min_lon < df.lon, df.lon < max_lon), inplace=True)
df.dropna(inplace=True)

# parse datetimes in snuffelfiets data
if parse_dates:
    df.receive_time = df.receive_time.apply(datetime.datetime.strptime, args=('%Y-%m-%d %H:%M:%S',))

# output amount of entries matching filters
print('---------------------------------------------------------')
print('Total amount of records after filtering:', df.shape[0])

# print correlation cross table for particulate matter
print('---------------------------------------------------------')
print('Particulate sizes correclations:')
print(df[['pm10', 'pm2_5', 'pm1_0']].corr())

# get descriptive statistics for particulate matter levels
descriptives = df[['pm10', 'pm2_5', 'pm1_0']].describe()
print('---------------------------------------------------------')
print('Descriptive statistics for particulate matter:')
print(descriptives)

# get daily averages for particulate matter for use in simplified model
df_copy = df[['receive_time', 'pm10', 'pm2_5', 'pm1_0']].copy(deep=True)
df_copy = df_copy.resample('1D', on='receive_time').mean()
df_copy.dropna(inplace=True)
df_weather_copy = df_weather.copy(deep=True)
df_weather_copy.where(df_weather_copy.timestamp.isin(df_copy.index), inplace=True)
df_weather_copy.dropna(inplace=True)

# split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(df_weather_copy[['wind_x', 'wind_y']],
                                                    df_copy['pm2_5'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    )

# fit linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# calculate regression accuracy measures
r_square = reg.score(x_test, y_test)
y_predict = reg.predict(x_test)
rms_error = mean_squared_error(y_predict, y_test, squared=False)

# print regression accuracy measures
print('---------------------------------------------------------')
print('Wind-Particulate regression model accuracy:')
print('R squared value:', r_square)
print('RMS error:', rms_error)

# get quartile indices
pm10 = np.array(list(df.pm10))
pm10_q1 = np.where(pm10 <= descriptives.pm10['25%'])
pm10_q2 = np.where(np.logical_and(pm10 > descriptives.pm10['25%'], pm10 <= descriptives.pm10['50%']))
pm10_q3 = np.where(np.logical_and(pm10 > descriptives.pm10['50%'], pm10 <= descriptives.pm10['75%']))
pm10_q4 = np.where(pm10 > descriptives.pm10['75%'])

pm2_5 = np.array(list(df.pm2_5))
pm2_5_q1 = np.where(pm2_5 <= descriptives.pm2_5['25%'])
pm2_5_q2 = np.where(np.logical_and(pm2_5 > descriptives.pm2_5['25%'], pm2_5 <= descriptives.pm2_5['50%']))
pm2_5_q3 = np.where(np.logical_and(pm2_5 > descriptives.pm2_5['50%'], pm2_5 <= descriptives.pm2_5['75%']))
pm2_5_q4 = np.where(pm2_5 > descriptives.pm2_5['75%'])

pm1_0 = np.array(list(df.pm1_0))
pm1_0_q1 = np.where(pm1_0 <= descriptives.pm1_0['25%'])
pm1_0_q2 = np.where(np.logical_and(pm1_0 > descriptives.pm1_0['25%'], pm1_0 <= descriptives.pm1_0['50%']))
pm1_0_q3 = np.where(np.logical_and(pm1_0 > descriptives.pm1_0['50%'], pm1_0 <= descriptives.pm1_0['75%']))
pm1_0_q4 = np.where(pm1_0 > descriptives.pm1_0['75%'])

# get experiment duration
start_time = df.receive_time.min()
end_time = df.receive_time.max()
experiment_duration = end_time - start_time
experiment_duration = experiment_duration.days

# for now, use numpy arrays for plot data
time = np.array(list(df.receive_time))
lon = np.array(list(df.lon))
lat = np.array(list(df.lat))

time_weather = np.array(list(df_weather.timestamp))
wind_x = np.array(list(df_weather.wind_x))
wind_y = np.array(list(df_weather.wind_y))

# get initial time slice
initial_time_slice = np.where(np.logical_and(time >= start_time, time <= start_time + interval_duration))
initial_time_slice_weather = np.where(np.logical_and(time_weather >= start_time, time_weather <= start_time + interval_duration))

# read map image
map_image = plt.imread('map_sittard_geleen.png')

# create plot window
fig = plt.figure()
# specify grid for plotting window
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
# axis 1: cartesian axis for coordinate plotting
ax1 = plt.subplot(gs[0])
# axis 2: polar axis for wind direction plotting
ax2 = plt.subplot(gs[1], projection='polar')


# create scatter plots and plot initial coordinates
quartile_1, = ax1.plot(lon[np.intersect1d(pm2_5_q1, initial_time_slice)],
                       lat[np.intersect1d(pm2_5_q1, initial_time_slice)],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='g', label='pm2_5 1st quartile')
quartile_2, = ax1.plot(lon[np.intersect1d(pm2_5_q2, initial_time_slice)],
                       lat[np.intersect1d(pm2_5_q2, initial_time_slice)],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='y', label='pm2_5 2nd quartile')
quartile_3, = ax1.plot(lon[np.intersect1d(pm2_5_q3, initial_time_slice)],
                       lat[np.intersect1d(pm2_5_q3, initial_time_slice)],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='r', label='pm2_5 3rd quartile')
quartile_4, = ax1.plot(lon[np.intersect1d(pm2_5_q4, initial_time_slice)],
                       lat[np.intersect1d(pm2_5_q4, initial_time_slice)],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='k', label='pm2_5 4th quartile')


# helper function to convert cartesian vector coordinates to polar
def cart2pol(x, y):
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius


# convert wind data to polar coordinates
angles, radii = cart2pol(wind_x[initial_time_slice_weather],
                         wind_y[initial_time_slice_weather])
# annotation list to keep track of annotations on wind graph
annotations = []
# set polar plot border to maximum plot value
ax2.set_ylim(0, np.max(radii)/10)
# use annotations to indicate wind direction and speed on polar graph, keep track of annotations in annotations list
for angle, radius in zip(angles, radii):
    ann = ax2.annotate("", xy=(angle, radius/10), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='b'))
    annotations.append(ann)
# add a graph title to wind graph
ax2.set_title('wind [m/s]')


# Create slider to shift time window
slider_axes = plt.axes([0.25, 0.1, 0.65, 0.03])
time_offset_slider = Slider(
    ax=slider_axes,
    label='Time Offset',
    valmin=0,
    valmax=experiment_duration - interval_width,
    valinit=0,
)


# Create save button
button_axes = plt.axes([0.8, 0.025, 0.1, 0.04])
save_button = Button(button_axes, 'Save', hovercolor='0.975')


# On changed callback for slider
def update_plot(val):
    # get time offset value from slider widget
    offset = time_offset_slider.val
    # convert offset to datetime compatible duration
    offset_duration = datetime.timedelta(days=offset)

    # get selected slice of data for snuffelfiets data
    time_slice = np.where(np.logical_and(time >= start_time + offset_duration,
                                         time <= start_time + offset_duration + interval_duration))

    # get selected slice of data for weather data
    time_slice_weather = np.where(np.logical_and(time_weather >= start_time + offset_duration,
                                                 time_weather <= start_time + offset_duration + interval_duration))

    # remove previous wind annotations (if this step is skipped, previous indicators will remain onscreen)
    for annotation in annotations:
        annotation.remove()
    # reset annotation tracking list
    annotations.clear()

    # convert selected slice of vector wind data to polar coordinates
    angles, radii = cart2pol(wind_x[time_slice_weather],
                             wind_y[time_slice_weather])
    # set border for polar plot to maximum plot value
    ax2.set_ylim(0, np.max(radii)/10)

    # display wind direction and speed using annotations
    for angle, radius in zip(angles, radii):
        ann = ax2.annotate("", xy=(angle, radius/10), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='b'))
        # track annotations in annotations list
        annotations.append(ann)

    # swap out data for the scatter plots to the new selection
    quartile_1.set_data(lon[np.intersect1d(pm2_5_q1, time_slice)], lat[np.intersect1d(pm2_5_q1, time_slice)])
    quartile_2.set_data(lon[np.intersect1d(pm2_5_q2, time_slice)], lat[np.intersect1d(pm2_5_q2, time_slice)])
    quartile_3.set_data(lon[np.intersect1d(pm2_5_q3, time_slice)], lat[np.intersect1d(pm2_5_q3, time_slice)])
    quartile_4.set_data(lon[np.intersect1d(pm2_5_q4, time_slice)], lat[np.intersect1d(pm2_5_q4, time_slice)])

    # redraw the canvas
    fig.canvas.draw_idle()


# Save function for save button
def save_plot(event):
    # TODO: add save function
    print('SAVING NOT YET IMPLEMENTED.')


# register slider- and button callbacks
time_offset_slider.on_changed(update_plot)
save_button.on_clicked(save_plot)

# set axis limits
ax1.set_xlim(map_bounds[0], map_bounds[1])
ax1.set_ylim(map_bounds[2], map_bounds[3])

# set coordinate graph title
ax1.set_title('Particulate matter in municipality Sittard-Geleen.')

# set axis titles
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# enable legend
ax1.legend(loc='upper right')

# plot image
ax1.imshow(map_image, zorder=0, extent=map_bounds, aspect='equal')

# show plot window
plt.show()
