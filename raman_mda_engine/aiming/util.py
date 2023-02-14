from math import floor

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def polygon_laser_focus(shape_data, shape_type, density, plot=True):
    """
    shape_data is the array of points that napari uses to define the shape
    shape_type is the type the napari assigns to the shape
    density is the pixels per interval between lattice points.
    """

    def rectangle(rect, d_r):
        h, w = abs(rect[2, 0] - rect[1, 0]), abs(rect[1, 1] - rect[0, 1])
        if h > w:
            n_r = int((rect[2, 0] - rect[1, 0]) / d_r)
            heights = np.linspace(rect[1, 0], rect[2, 0], n_r)
            n_h = floor(w / (h / (n_r - 1)))
            if n_h != 0:
                widths = np.linspace(rect[0, 1], rect[1, 1], n_h + 2)
            else:
                widths = np.array([rect[0, 1], rect[1, 1]])

        else:
            n_r = int((rect[1, 1] - rect[0, 1]) / d_r)
            widths = np.linspace(rect[0, 1], rect[1, 1], n_r)
            n_w = floor(h / (w / (n_r - 1)))
            if n_w != 0:
                heights = np.linspace(rect[1, 0], rect[2, 0], n_w + 2)
            else:
                heights = np.array([rect[1, 0], rect[2, 0]])

        points = []
        for width in widths:
            for height in heights:
                points.append([height, width])

        return np.array(points)

    def circle(circ, d_c):
        (circ[2, 0] - circ[0, 0]) / 2
        points = []
        y_cm, x_cm = (circ[2, 0] + circ[0, 0]) / 2, (circ[1, 1] + circ[0, 1]) / 2
        rady, radx = abs(circ[2, 0] - circ[0, 0]) / 2, abs(circ[1, 1] - circ[0, 1]) / 2
        if radx > rady:
            n_c = int(abs(circ[0, 1] - circ[1, 1]) / d_c)
            rxs = np.linspace(circ[1, 1], circ[0, 1], n_c)
            for i, rx in enumerate(rxs):
                if i == 0 or i == len(rxs) - 1:
                    curr_y = y_cm
                else:
                    curr_y = rady * np.sqrt(1 - (rx - x_cm) ** 2 / radx**2) + y_cm

                n_y = floor((curr_y - y_cm) / (radx / (n_c - 1)))
                rys = np.linspace(2 * y_cm - curr_y, curr_y, n_y + 2)
                for ry in rys:
                    points.append([ry, rx])
        else:
            n_c = int(abs(circ[2, 0] - circ[0, 0]) / d_c)
            rys = np.linspace(circ[0, 0], circ[2, 0], n_c)
            for j, ry in enumerate(rys):
                if j == 0 or j == len(rys) - 1:
                    curr_x = x_cm
                else:
                    curr_x = radx * np.sqrt(1 - (ry - y_cm) ** 2 / rady**2) + x_cm

                n_x = floor((curr_x - x_cm) / (rady / (n_c - 1)))
                rxs = np.linspace(2 * x_cm - curr_x, curr_x, n_x + 2)
                for rx in rxs:
                    points.append([ry, rx])

        return np.array(points)

    def irregular(irr, d_i):
        y_min, y_max, x_min, x_max = (
            min(irr[:, 1]),
            max(irr[:, 1]),
            min(irr[:, 0]),
            max(irr[:, 0]),
        )
        rect_points = rectangle(
            np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]),
            d_i,
        )
        polygon = Polygon(irr)
        points = []
        for rect_point in rect_points:
            P = Point(rect_point[0], rect_point[1])
            if polygon.contains(P) is True:
                points.append(rect_point)

        return np.array(points)

    if shape_type == "rectangle":
        points = rectangle(shape_data, density)
    elif shape_type == "ellipse":
        points = circle(shape_data, density)
    elif shape_type == "polygon":
        points = irregular(shape_data, density)

    if plot:
        plt.figure()
        plt.scatter(points.T[0], points.T[1])
        plt.scatter(shape_data.T[0], shape_data.T[1])
        plt.axis("scaled")

    # return points
    return points


def brush_laser_focus(label_data, density, plot=True):
    """
    shape_data is the array of points that napari uses to define the shape
    shape_type is the type the napari assigns to the shape
    density is the pixels per interval between lattice points.
    """

    def rectangle(rect, d_r):
        h, w = abs(rect[2, 0] - rect[1, 0]), abs(rect[1, 1] - rect[0, 1])
        if h > w:
            n_r = int((rect[2, 0] - rect[1, 0]) / d_r)
            heights = np.linspace(rect[1, 0], rect[2, 0], n_r)
            n_h = floor(w / (h / (n_r - 1)))
            if n_h != 0:
                widths = np.linspace(rect[0, 1], rect[1, 1], n_h + 2)
            else:
                widths = np.array([rect[0, 1], rect[1, 1]])

        else:
            n_r = int((rect[1, 1] - rect[0, 1]) / d_r)
            widths = np.linspace(rect[0, 1], rect[1, 1], n_r)
            n_w = floor(h / (w / (n_r - 1)))
            if n_w != 0:
                heights = np.linspace(rect[1, 0], rect[2, 0], n_w + 2)
            else:
                heights = np.array([rect[1, 0], rect[2, 0]])

        points = []
        for width in widths:
            for height in heights:
                points.append([height, width])

        return np.array(points)

    ones = np.ones(label_data.shape[0])
    y_min, y_max = (
        np.nonzero(np.matrix(label_data) @ ones)[1][0],
        np.nonzero(np.matrix(label_data) @ ones)[1][-1],
    )
    x_min, x_max = (
        np.nonzero(np.matrix(label_data).T @ ones)[1][0],
        np.nonzero(np.matrix(label_data).T @ ones)[1][-1],
    )

    rect = np.rint(
        rectangle(
            np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]),
            density,
        )
    )
    points = rect[label_data[rect[:, 1].astype(int), rect[:, 0].astype(int)] > 0][
        :, ::-1
    ]

    if plot:
        plt.figure()
        plt.imshow(label_data.T)
        plt.scatter(points.T[0], points.T[1])
        plt.axis("scaled")

    return points
