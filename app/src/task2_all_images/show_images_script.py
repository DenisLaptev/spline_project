import os
import re  # for reg exp
import cv2
import numpy as np
import matplotlib.pyplot as plt




# # popt:array=Optimal values for the parameters so that the sum
# # of the squared residuals of f(xdata, *popt) - ydata is minimized
#
# # pcov:2d array=The estimated covariance of popt.
# # The diagonals provide the variance of the parameter estimate.


# height of image
HEIGHT = 360

# width of image
WIDTH = 640


def f3(y, a, b, c, d):
    return a + b * y + c * y ** 2 + d * y ** 3


def f4(y, a, b, c, d, e):
    return a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4


def poly2(y_list, a, b, c):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2)

    # print("x_list=", x_list)
    # print("y_list=", y_list)

    return x_list


def poly3(y_list, a, b, c, d):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3)

    # print("x_list=", x_list)
    # print("y_list=", y_list)

    return x_list


def poly4(y_list, a, b, c, d, e):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4)

    # print("x_list=", x_list)
    # print("y_list=", y_list)

    return x_list


def poly5(y_list, a, b, c, d, e, f):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4 + f * y ** 5)

    # print("x_list=", x_list)
    # print("y_list=", y_list)

    return x_list


def ys_opencv_to_ys_matplotlib(ys_opencv, height):
    ys_matplotlib = []
    for i in range(len(ys_opencv)):
        ys_matplotlib.append(height - ys_opencv[i])
    return ys_matplotlib


def ys_matplotlib_to_ys_opencv(ys_matplotlib, height):
    ys_opencv = []
    for i in range(len(ys_matplotlib)):
        ys_opencv.append(height - ys_matplotlib[i])
    return ys_opencv


def generate_xs_ys_lists_from_xy_list(xy_list):
    # print("xy_list=", xy_list)
    xs = []
    ys = []
    yps = []
    for i in range(len(xy_list)):
        if i % 2 == 0:
            xs.append(xy_list[i])
        elif i % 2 == 1:
            ys.append(xy_list[i])

    yps = ys_opencv_to_ys_matplotlib(ys_opencv=ys, height=HEIGHT)

    return xs, yps


def get_lines_list_from_file(path_to_file):
    file = open(path_to_file, "r")
    lines = file.read().split("\n")
    for line in lines:
        line.strip()
    # print(file.readlines())
    # print("len(lines)=", len(lines))
    file.close()
    return lines


def get_numbers_list_from_line(line):
    numbers_list = []
    elements = line.split(" ")
    for element in elements:
        if len(element) != 0:
            numbers_list.append(int(element))
    # print("numbers_list=", numbers_list)
    return numbers_list


def get_xs_ys_lists_from_straight(x1, x2, y1, y2):
    xs = []
    ys = []

    if y1 <= y2:
        number_of_points = y2 - y1
        for i in range(number_of_points):
            y = y1 + i
            x = x1 + (x2 - x1) / (y2 - y1) * (y - y1)
            ys.append(y)
            xs.append(x)

    elif y1 > y2:
        number_of_points = y1 - y2
        for i in range(number_of_points):
            y = y1 - i
            x = x1 + (x2 - x1) / (y2 - y1) * (y - y1)
            ys.append(y)
            xs.append(x)

    # print("xs_straight=", xs)
    # print("ys_straight=", ys)

    return xs, ys


def get_xs_ys_lists_from_curve(poly_order, popt, y1, y2):
    xs = []
    ys = []

    if poly_order == 'poly3':
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]

    elif poly_order == 'poly4':
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        e = popt[4]

    if y1 <= y2:
        number_of_points = y2 - y1
        for i in range(number_of_points):
            y = y1 + i
            if poly_order == 'poly3':
                x = f3(y, a, b, c, d)
            elif poly_order == 'poly4':
                x = f4(y, a, b, c, d, e)
            ys.append(y)
            xs.append(x)

    elif y1 > y2:
        number_of_points = y1 - y2
        for i in range(number_of_points):
            y = y1 - i
            if poly_order == 'poly3':
                x = f3(y, a, b, c, d)
            elif poly_order == 'poly4':
                x = f4(y, a, b, c, d, e)
            ys.append(y)
            xs.append(x)

    # print("xs_curve=", xs)
    # print("ys_curve=", ys)

    return xs, ys


def get_data_lists_from_folder(path_to_folder):
    paths_to_files = []
    file_names = []
    frame_numbers = []
    # r = root, d = directories, f = files
    for r, d, f in os.walk(path_to_folder):
        for file in f:
            if '.txt' in file:
                path_to_file = os.path.join(r, file)
                paths_to_files.append(path_to_file)
                file_names.append(file)

    print('-----paths_to_files------')

    for f in paths_to_files:
        print(f)

    print('-----file_names------')

    for fn in file_names:
        print(fn)

    print('-----frame_numbers------')

    for fn in file_names:
        for m in re.finditer(r"(.+)(_frame_)(\w+)(\.lines)", fn):
            frame_numbers.append(int(m.group(3)))

    for frame_number in frame_numbers:
        print(frame_number)

    return frame_numbers, file_names, paths_to_files


frame_numbers, file_names, paths_to_files = get_data_lists_from_folder(path_to_folder='../Smoothed/')

for i in range(len(paths_to_files)):
    frame=frame_numbers[i]
    path=paths_to_files[i]
    lines = get_lines_list_from_file(path_to_file=path)

    # ---------------------------------initial_figure----------------------------
    title_of_figure = "FRAME_" + str(frame)
    plt.figure(figsize=(20, 10))
    plt.suptitle(title_of_figure)

    if len(lines) >=1:
        xs1, yps1 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[0]))
        plt.plot(xs1, yps1)

    if len(lines) >=2:
        xs2, yps2 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[1]))
        plt.plot(xs2, yps2)

    if len(lines) >=3:
        xs3, yps3 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[2]))
        plt.plot(xs3, yps3)

    if len(lines) >=4:
        xs4, yps4 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[3]))
        plt.plot(xs4, yps4)


    plt.xlim((0, 640))
    plt.ylim((0, 360))
    plt.xlabel('x')
    plt.ylabel('y')
    path_to_file_with_plots_name = './initial_frame' + str(frame) + '.png'
    #plt.savefig(path_to_file_with_plots_name)
    plt.show()
