import numpy as np
import matplotlib.pyplot as plt
from annotation import Arrow3D


class CylinderPlot:
    @staticmethod
    def cylindrical_plot(mapping_array, title=None, radius=1, height_z=1, theta_resolution=360):
        # height and angle range
        z = np.linspace(0, height_z, theta_resolution)
        theta = np.linspace(0, -2*np.pi, theta_resolution)

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
        # ax.annotate3D('N', (1, 0, 0),  xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", ec='red', fc='white'))
        ax.arrow3D(0, 0, 0, 1, 0, 0, arrowstyle="-|>", ec='red', fc='white', mutation_scale=20)
        ax.arrow3D(0, 0, 0, -1, 0, 0, arrowstyle="-|>", ec='black', fc='white', mutation_scale=20)
        if title is not None:
            ax.set_title(title)
        plt.show()
        return


# predict_wind_direction = np.linspace(1, 360, 360)
# transformation_array = predict_wind_direction/360
#
# CylinderPlot.cylindrical_plot((transformation_array-np.mean(transformation_array))/np.mean(transformation_array))
# # CylinderPlot.cylindrical_plot(transformation_array)