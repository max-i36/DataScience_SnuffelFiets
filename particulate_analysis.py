import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from error_unpacker import ErrorUnpacker
from matplotlib.widgets import Slider, Button
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from custom_plotting_tools import CustomPlottingTools, ConversionTools
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec

# parse_dates = False
parse_dates = True

# import snuffelfiets csv data to pandas dataframe
df = pd.read_csv('snuffelfiets_data_filtered.csv')

# import weather data to dataframe
df_weather = pd.read_csv('weerdata.csv')
# only keep relevant columns to save memory
df_weather = df_weather[['YYYYMMDD', 'DDVEC', 'FHVEC', '   PG']]
# rename columns to something more human-readable
df_weather.rename(columns={'YYYYMMDD': 'timestamp',
                           'DDVEC': 'wind_direction',
                           'FHVEC': 'wind_speed',
                           '   PG': 'air_pressure',
                           },
                  inplace=True,
                  )

# get wind vector components
wind_vectors = ConversionTools.vector_wind(df_weather.wind_direction, df_weather.wind_speed)

# add wind vector components to weather dataframe
df_weather = pd.concat([df_weather, wind_vectors], axis=1)

# parse timestamp column using custom function
df_weather.timestamp = df_weather.timestamp.apply(ConversionTools.int_to_timestamp)

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

# coordinate grid
lon_vector = np.linspace(min_lon, max_lon, 200)
lat_vector = np.linspace(min_lat, max_lat, 200)

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

# split into training and testing data for multiple linear regression
x_train, x_test, y_train, y_test = train_test_split(df_weather_copy[['wind_x', 'wind_y']],
                                                    df_copy['pm2_5'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    )

# fit linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# fit neural network regression model
neural_net_wind = MLPRegressor(max_iter=200,
                               solver='sgd',
                               activation='logistic',
                               learning_rate_init=0.01,
                               early_stopping=False,
                               random_state=42,
                               )
neural_net_wind.fit(x_train, y_train)

# calculate regression accuracy measures
# linear
y_predict = reg.predict(x_test)
r_square = r2_score(y_test, y_predict)
rms_error = mean_squared_error(y_predict, y_test, squared=False)
# neural net
y_predict_neural = neural_net_wind.predict(x_test)
r_square_neural = r2_score(y_test, y_predict_neural)
rms_error_neural = mean_squared_error(y_predict_neural, y_test, squared=False)

# print regression accuracy measures
print('---------------------------------------------------------')
print('Wind-Particulate linear regression model accuracy:')
print('R squared value (linear):', r_square)
print('RMS error (linear):', rms_error)
print('Wind-Particulate neural net regression model accuracy:')
print('R squared value (neural):', r_square_neural)
print('RMS error (neural):', rms_error_neural)

# # display regression results visually # #
# set up input wind data to use in prediction model (mean wind speed, all directions covered)
predict_wind_speed = np.ones(72) * df_weather.wind_speed.mean()
predict_wind_direction = np.linspace(5, 360, 72)
# set up dataframe
df_predict = pd.DataFrame({'wind_speed': predict_wind_speed, 'wind_direction': predict_wind_direction})
# vectorize wind for prediction input
wind_vectors_predict = ConversionTools.vector_wind(df_predict.wind_direction, df_predict.wind_speed)
# generate prediction (linear)
transformation_array = reg.predict(wind_vectors_predict)
# generate prediction (neural net)
transformation_array_neural = neural_net_wind.predict(wind_vectors_predict)
# use mean to transform prediction to relative in- and decreases (linear)
relative_transformation_array = (transformation_array - np.mean(transformation_array))/np.mean(transformation_array)
# use mean to transform prediction to relative in- and decreases (neural net)
relative_transformation_array_neural = (transformation_array_neural -
                                        np.mean(transformation_array_neural))/np.mean(transformation_array_neural)

# create cylinder plot (linear)
CustomPlottingTools.cylindrical_plot(relative_transformation_array,
                                     title='Relative particulate increase in function of wind direction\nNOTE: wind '
                                           'direction in meteorological notation.\nI.E. where wind is blowing FROM. '
                                           '(linear)',
                                     theta_resolution=72,
                                     )

# create cylinder plot (neural net)
CustomPlottingTools.cylindrical_plot(transformation_array_neural,
                                     title='Relative particulate increase in function of wind direction\nNOTE: wind '
                                           'direction in meteorological notation.\nI.E. where wind is blowing FROM. '
                                           '(neural)',
                                     theta_resolution=72,
                                     )

# create rose plot (linear)
CustomPlottingTools.rose_plot(predict_wind_direction,
                              transformation_array,
                              title='Average particulate level in function of wind direction (linear prediction). '
                                    '[mg/m^3]',
                              cm='rainbow'
                              )

# create rose plot (neural net)
CustomPlottingTools.rose_plot(predict_wind_direction,
                              transformation_array_neural,
                              title='Average particulate level in function of wind direction (neural net prediction). '
                                    '[mg/m^3]',
                              cm='rainbow'
                              )

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
initial_time_slice_weather = np.where(np.logical_and(time_weather >= start_time,
                                                     time_weather <= start_time + interval_duration,
                                                     ))

# create dictionary for wind vectors with dates as keys
wind_dict = {}
for date, wind_x_val, wind_y_val in zip(list(df_weather.timestamp), list(df_weather.wind_x), list(df_weather.wind_y)):
    wind_dict[date.date()] = (wind_x_val, wind_y_val)

# set up column vector of pm2_5 quartiles
pm2_5_quartiles = np.ones(len(df.index))
pm2_5_quartiles[pm2_5_q2] *= 2
pm2_5_quartiles[pm2_5_q3] *= 3
pm2_5_quartiles[pm2_5_q4] *= 4

# set up column vectors containing wind data appropriate for the days in the main dataframe
wind_x_vector = []
wind_y_vector = []
times = list(df.receive_time)
for i in range(0, len(times)):
    wind_x_vector.append(wind_dict[times[i].date()][0])
    wind_y_vector.append(wind_dict[times[i].date()][1])

# add wind data to the dataframe
df['wind_x'] = wind_x_vector
df['wind_y'] = wind_y_vector
df['pm2_5_quartiles'] = pm2_5_quartiles

df['pm2_5_quartiles'] = LabelEncoder().fit_transform(df['pm2_5_quartiles'])

# split training and test data for neural net
x_train, x_test, y_train, y_test = train_test_split(df[['wind_x', 'wind_y', 'lon', 'lat']],
                                                    df['pm2_5_quartiles'],
                                                    test_size=0.1,
                                                    random_state=36,
                                                    )
# # fit neural network classifier model
neural_net = MLPClassifier(max_iter=200,
                           solver='sgd',
                           activation='logistic',
                           learning_rate_init=0.01,
                           early_stopping=True,
                           )
neural_net.fit(x_train, y_train)
print('Neural net accuracy: ', neural_net.score(x_test, y_test))

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


# convert wind data to polar coordinates
angles, radii = ConversionTools.cart2pol(wind_x[initial_time_slice_weather],
                                         wind_y[initial_time_slice_weather],
                                         )
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
    angles, radii = ConversionTools.cart2pol(wind_x[time_slice_weather],
                                             wind_y[time_slice_weather],
                                             )
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

#######################################################################################################################

# make prediction using neural net classifier
longs_predict = []
lats_predict = []
for i in range(0, len(lon_vector)):
    for j in range(0, len(lat_vector)):
        longs_predict.append(lon_vector[i])
        lats_predict.append(lat_vector[j])

# enter wind direction (cartesian)
wind_x_predict = 0
wind_y_predict = 0

# create array for wind direction
wind_x_predict_array = np.ones(len(longs_predict)) * wind_x_predict
wind_y_predict_array = np.ones(len(longs_predict)) * wind_y_predict

# set up input dataframe for predictions
df_coords_predict = pd.DataFrame({'wind_x': wind_x_predict_array,
                                  'wind_y': wind_y_predict_array,
                                  'lon': longs_predict,
                                  'lat': lats_predict,
                                  })

# predict particulate levels
neural_net_prediction = neural_net.predict(df_coords_predict)
# isolate prediction from dataframe
df_coords_predict['neural_predict'] = neural_net_prediction

# split prediction into quartile values
pred_q1 = np.where(neural_net_prediction == 0)
pred_q2 = np.where(neural_net_prediction == 1)
pred_q3 = np.where(neural_net_prediction == 2)
pred_q4 = np.where(neural_net_prediction == 3)

# get gps coordinate arrays for prediction
lon = np.array(list(df_coords_predict.lon))
lat = np.array(list(df_coords_predict.lat))

# create plot window
fig = plt.figure()
# specify grid for plotting window
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
# axis 1: cartesian axis for coordinate plotting
ax1 = plt.subplot(gs[0])
# axis 2: polar axis for wind direction plotting
ax2 = plt.subplot(gs[1], projection='polar')

# set axis limits
ax1.set_xlim(map_bounds[0], map_bounds[1])
ax1.set_ylim(map_bounds[2], map_bounds[3])

# set coordinate graph title
ax1.set_title('Particulate matter in municipality Sittard-Geleen.')

# create scatter plots and plot initial coordinates
quartile_1, = ax1.plot(lon[pred_q1],
                       lat[pred_q1],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='g', label='pm2_5 1st quartile')
quartile_2, = ax1.plot(lon[pred_q2],
                       lat[pred_q2],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='y', label='pm2_5 2nd quartile')
quartile_3, = ax1.plot(lon[pred_q3],
                       lat[pred_q3],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='r', label='pm2_5 3rd quartile')
quartile_4, = ax1.plot(lon[pred_q4],
                       lat[pred_q4],
                       zorder=1, marker='.', linestyle='', alpha=0.2, c='k', label='pm2_5 4th quartile')

# convert wind data to polar coordinates
angle, radius = ConversionTools.cart2pol(0,
                                         0,
                                         )

# annotation list to keep track of annotations on wind graph
annotations = []
# set polar plot border to maximum plot value
ax2.set_ylim(0, 30)
# use annotations to indicate wind direction and speed on polar graph, keep track of annotations in annotations list
ann = ax2.annotate("", xy=(ConversionTools.deg2rad(angle), radius), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='b'))
annotations.append(ann)
# add a graph title to wind graph
ax2.set_title('wind [m/s]')

# read map image
map_image = plt.imread('map_sittard_geleen.png')

# Create slider to control wind direction
slider_x = plt.axes([0.25, 0.04, 0.65, 0.02])
x_slider = Slider(
    ax=slider_x,
    label='wind x',
    valmin=-20,
    valmax=20,
    valinit=0,
)

# Create slider to control wind speed
slider_y = plt.axes([0.04, 0.25, 0.02, 0.63])
y_slider = Slider(
    ax=slider_y,
    label="wind y",
    valmin=-20,
    valmax=20,
    valinit=0,
    orientation="vertical"
)

ax2.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])


# On changed callback for slider
def update_plot(val):
    # remove previous wind annotations (if this step is skipped, previous indicators will remain onscreen)
    for annotation in annotations:
        annotation.remove()
    # reset annotation tracking list
    annotations.clear()

    # get wind settings from sliders
    x = x_slider.val
    y = y_slider.val

    # convert cartesian wind data to polar
    angle, radius = ConversionTools.cart2pol(x, y)

    # draw annotation arrow on polar plot
    ann = ax2.annotate("", xy=(angle, radius), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='b'))
    # track annotations in annotations list
    annotations.append(ann)

    # create array for wind direction
    wind_x_predict_array = np.ones(len(longs_predict)) * x
    wind_y_predict_array = np.ones(len(longs_predict)) * y

    # set up input dataframe for predictions
    df_coords_predict = pd.DataFrame({'wind_x': wind_x_predict_array,
                                      'wind_y': wind_y_predict_array,
                                      'lon': longs_predict,
                                      'lat': lats_predict,
                                      })

    # predict particulate levels
    neural_net_prediction = neural_net.predict(df_coords_predict)
    # isolate prediction from dataframe
    df_coords_predict['neural_predict'] = neural_net_prediction

    # split prediction into quartile values
    pred_q1 = np.where(neural_net_prediction == 0)
    pred_q2 = np.where(neural_net_prediction == 1)
    pred_q3 = np.where(neural_net_prediction == 2)
    pred_q4 = np.where(neural_net_prediction == 3)

    # get gps coordinate arrays for prediction
    lon = np.array(list(df_coords_predict.lon))
    lat = np.array(list(df_coords_predict.lat))

    # swap out data for the scatter plots to the new selection
    quartile_1.set_data(lon[pred_q1], lat[pred_q1])
    quartile_2.set_data(lon[pred_q2], lat[pred_q2])
    quartile_3.set_data(lon[pred_q3], lat[pred_q3])
    quartile_4.set_data(lon[pred_q4], lat[pred_q4])

    # redraw the canvas
    fig.canvas.draw_idle()


# register slider- and button callbacks
x_slider.on_changed(update_plot)
y_slider.on_changed(update_plot)

# set axis titles
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# enable legend
ax1.legend(loc='upper right')

# plot image
ax1.imshow(map_image, zorder=0, extent=map_bounds, aspect='equal')

plt.show()
