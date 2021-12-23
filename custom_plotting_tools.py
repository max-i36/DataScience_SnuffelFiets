import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
from annotation import Arrow3D


class ConversionTools:
    @staticmethod
    # function to convert cartesian coordinates to polar coordinates
    def cart2pol(x, y):
        radius = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return theta, radius

    @staticmethod
    # function to convert angles in degrees to radians
    def deg2rad(degrees):
        return (degrees * 2 * math.pi) / 360

    @staticmethod
    # function to convert YYYYMMDD formatted integers to datetime objects
    def int_to_timestamp(integer):
        string = str(integer)
        return datetime.datetime.strptime(string, '%Y%m%d')

    @staticmethod
    # function to decompose wind information from meteorological degrees to vector components
    def vector_wind(wind_direction, wind_speed):
        # convert direction to radians
        wind_direction = wind_direction / 180 * math.pi
        # get x and y components
        wind_x = pd.DataFrame(-wind_speed.abs() * wind_direction.apply(math.sin), columns=['wind_x'])
        wind_y = pd.DataFrame(-wind_speed.abs() * wind_direction.apply(math.cos), columns=['wind_y'])
        return pd.concat([wind_x, wind_y], axis=1)


class CustomPlottingTools:
    @staticmethod
    def cylindrical_plot(mapping_array, title=None, radius=1, height_z=1, theta_resolution=360):
        # height and angle range
        z = np.linspace(0, height_z, theta_resolution)
        theta = np.linspace(0, -2 * np.pi, theta_resolution)

        # set up mesh grid
        theta_grid, z_grid = np.meshgrid(theta, z)

        # combine theta with radius to get ciclular base for cyclinder in cartesian coordinates
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        # scale z values in mesh to morph cyclinder to input mapping array
        for row in z_grid:
            for i in range(0, len(row)):
                row[i] = row[i] * mapping_array[i]

        # create figure
        fig = plt.figure()
        # set axis mode to 3d
        ax = fig.add_subplot(111, projection='3d')

        # plot the meshed surface
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=1)
        ax.arrow3D(0, 0, 0, 1, 0, 0, arrowstyle="-|>", ec='red', fc='white', mutation_scale=20)
        ax.arrow3D(0, 0, 0, -1, 0, 0, arrowstyle="-|>", ec='black', fc='white', mutation_scale=20)
        if title is not None:
            ax.set_title(title)
        plt.show()
        return

    @staticmethod
    def arrow_plot(angles, radii, title=None, convert_radians=True):
        # create figure
        fig = plt.figure()

        # convert degrees to radians
        if convert_radians:
            angles = ConversionTools.deg2rad(angles)

        # create axes
        ax = plt.subplot(projection='polar')
        for angle, radius in zip(angles, radii):
            ax.annotate("", xy=(angle, radius), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='b'))

        # set correct limit
        ax.set_ylim(0, np.max(radii) + 0.1 * np.max(radii))

        # set tick marks to geographical directions
        ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])

        # set graph title
        if title is not None:
            ax.set_title(title)

        # draw graph
        plt.show()
        return

    @staticmethod
    def rose_plot(angles, radii, title=None, convert_radians=True, cm=None):
        # create figure
        fig = plt.figure()

        # convert degrees to radians
        if convert_radians:
            angles = ConversionTools.deg2rad(angles)

        # create axes
        ax = plt.subplot(projection='polar')

        # appropriate plot for colormap
        width = ConversionTools.deg2rad(360 / len(angles))
        if cm is not None:
            # colormap selected
            colormap = plt.cm.get_cmap(cm)
            color = colormap((radii-np.min(radii))/(np.max(radii)-np.min(radii)))
            ax.bar(angles, radii, width=width, color=color)
        else:
            # no colormap, use default color
            ax.bar(angles, radii, width=width)

        # set correct limit
        ax.set_ylim(0, np.max(radii) + 0.1 * np.max(radii))

        # set tick marks to geographical directions
        ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])

        # set graph title
        if title is not None:
            ax.set_title(title)

        # draw graph
        plt.show()
        return


# # predict_wind_direction = np.linspace(10, 360, 36)
# predict_wind_direction = np.linspace(45, 360, 8)
# transformation_array = predict_wind_direction/360
# # transformation_array = predict_wind_direction
# # transformation_array = np.ones(36)
# # transformation_array = np.ones(8)
#
# # ArrowPlot.arrow_plot(predict_wind_direction, transformation_array, convert_radians=True)
#
# ArrowPlot.rose_plot(predict_wind_direction, transformation_array, cm='rainbow')
