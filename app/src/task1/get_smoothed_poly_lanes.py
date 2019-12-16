import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# popt:array=Optimal values for the parameters so that the sum
# of the squared residuals of f(xdata, *popt) - ydata is minimized

# pcov:2d array=The estimated covariance of popt.
# The diagonals provide the variance of the parameter estimate.

#Frames 239, 544, 594, 614, 877, 2305, 2316, 2616
FRAME_NUMBER = 2616
PATH_TO_INITIAL_FILE_Nagaraju = "../initial_images/_VL_Nagaraju_frame_" + str(FRAME_NUMBER) + ".lines.txt"
PATH_TO_INITIAL_FILE_ramarajuk = "../initial_images/_VL_ramarajuk_frame_" + str(FRAME_NUMBER) + ".lines.txt"

if FRAME_NUMBER in [239, 544, 594, 614, 877]:
    path_to_initial_file=PATH_TO_INITIAL_FILE_Nagaraju
elif FRAME_NUMBER in [2305, 2316, 2616]:
    path_to_initial_file = PATH_TO_INITIAL_FILE_ramarajuk



# number of points for approximation
N = 360

# height of image
HEIGHT = 360

# width of image
WIDTH = 640


def make_extended_coordinates_list(initial_coordinates_list, number_of_points):
    extended_coordinates_list = []
    ymin = np.min(initial_coordinates_list)
    ymax = np.max(initial_coordinates_list)
    delta = (ymax - ymin) / number_of_points
    for i in range(number_of_points):
        extended_coordinates_list.append(ymin + delta * i)
    return extended_coordinates_list


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


def generate_xs_ys_lists_from_xy_list(xy_list):
    print("xy_list=",xy_list)
    xs = []
    ys = []
    yps = []
    for i in range(len(xy_list)):
        if i % 2 == 0:
            xs.append(xy_list[i])
        elif i % 2 == 1:
            ys.append(xy_list[i])

    for i in range(len(ys)):
        yps.append(HEIGHT - ys[i])
    return xs, yps


def get_lines_list_from_file(frame_number,path_to_file):
    file = open(path_to_file, "r")
    lines = file.read().split("\n")
    for line in lines:
        line.strip()
    # print(file.readlines())
    print("len(lines)=", len(lines))
    file.close()
    return lines

def get_numbers_list_from_line(line):
    numbers_list=[]
    elements=line.split(" ")
    for element in elements:
        if len(element)!=0:
            numbers_list.append(int(element))
    print("numbers_list=",numbers_list)
    return numbers_list

lines = get_lines_list_from_file(frame_number=FRAME_NUMBER,
                                 path_to_file=path_to_initial_file)
xs1, yps1 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[0]))
xs2, yps2 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[1]))
xs3, yps3 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[2]))
if len(lines[3])!=0:
    xs4, yps4 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[3]))


# Fit for the parameters a, b, c of the function func
popt1_2, pcov1_2 = curve_fit(f=poly2,
                             xdata=yps1,
                             ydata=xs1)

popt2_2, pcov2_2 = curve_fit(f=poly2,
                             xdata=yps2,
                             ydata=xs2)

popt3_2, pcov3_2 = curve_fit(f=poly2,
                             xdata=yps3,
                             ydata=xs3)
if len(lines[3])!=0:
    popt4_2, pcov4_2 = curve_fit(f=poly2,
                                 xdata=yps4,
                                 ydata=xs4)
print("popt1_2=", popt1_2)
print("pcov1_2=", pcov1_2)

#########################################################

# Fit for the parameters a, b, c of the function func
popt1_3, pcov1_3 = curve_fit(f=poly3,
                             xdata=yps1,
                             ydata=xs1)

popt2_3, pcov2_3 = curve_fit(f=poly3,
                             xdata=yps2,
                             ydata=xs2)

popt3_3, pcov3_3 = curve_fit(f=poly3,
                             xdata=yps3,
                             ydata=xs3)
if len(lines[3])!=0:
    popt4_3, pcov4_3 = curve_fit(f=poly3,
                                 xdata=yps4,
                                 ydata=xs4)

print("popt1_3=", popt1_3)
print("pcov1_3=", pcov1_3)

#########################################################

# Fit for the parameters a, b, c of the function func
popt1_4, pcov1_4 = curve_fit(f=poly4,
                             xdata=yps1,
                             ydata=xs1)

popt2_4, pcov2_4 = curve_fit(f=poly4,
                             xdata=yps2,
                             ydata=xs2)

popt3_4, pcov3_4 = curve_fit(f=poly4,
                             xdata=yps3,
                             ydata=xs3)
if len(lines[3])!=0:
    popt4_4, pcov4_4 = curve_fit(f=poly4,
                                 xdata=yps4,
                                 ydata=xs4)
print("popt1_4=", popt1_4)
print("pcov1_4=", pcov1_4)

#########################################################

# Fit for the parameters a, b, c of the function func
popt1_5, pcov1_5 = curve_fit(f=poly5,
                             xdata=yps1,
                             ydata=xs1)

popt2_5, pcov2_5 = curve_fit(f=poly5,
                             xdata=yps2,
                             ydata=xs2)

popt3_5, pcov3_5 = curve_fit(f=poly5,
                             xdata=yps3,
                             ydata=xs3)
if len(lines[3])!=0:
    popt4_5, pcov4_5 = curve_fit(f=poly5,
                                 xdata=yps4,
                                 ydata=xs4)
print("popt1_5=", popt1_5)
print("pcov1_5=", pcov1_5)

title_of_figure = "FRAME_"+str(FRAME_NUMBER)
plt.suptitle(title_of_figure)
plt.plot(xs1, yps1)
plt.plot(xs2, yps2)
plt.plot(xs3, yps3)
if len(lines[3])!=0:
    plt.plot(xs4, yps4)
plt.xlim((0, 640))
plt.ylim((0, 360))
plt.xlabel('x')
plt.ylabel('y')
path_to_file_with_plots_name = './results_smoothed_lanes/initial_frame' + str(FRAME_NUMBER) + '.png'
plt.savefig(path_to_file_with_plots_name)

# fig=plt.figure()
# plt.rcParams['axes.facecolor'] = 'black'
plt.show()

title_of_figure = "FRAME_"+str(FRAME_NUMBER)
plt.figure(figsize=(20,10))
plt.suptitle(title_of_figure)
label2 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f'
plt.subplot(221)
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1, 'b-', label='data')
plt.plot(poly2(y_ext1, *popt1_2), y_ext1, 'r-', label=label2 % tuple(popt1_2))

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2, 'b-', label='data')
plt.plot(poly2(y_ext2, *popt2_2), y_ext2, 'r-', label=label2 % tuple(popt2_2))

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3, 'b-', label='data')
plt.plot(poly2(y_ext3, *popt3_2), y_ext3, 'r-', label=label2 % tuple(popt3_2))

if len(lines[3])!=0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4, 'b-', label='data')
    plt.plot(poly2(y_ext4, *popt4_2), y_ext4, 'r-', label=label2 % tuple(popt4_2))

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly2")
plt.xlabel('x')
plt.ylabel('y')

label3 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f,d=%5.3f'
plt.subplot(222)
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1, 'b-', label='data')
plt.plot(poly3(y_ext1, *popt1_3), y_ext1, 'r-', label=label3 % tuple(popt1_3))

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2, 'b-', label='data')
plt.plot(poly3(y_ext2, *popt2_3), y_ext2, 'r-', label=label3 % tuple(popt2_3))

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3, 'b-', label='data')
plt.plot(poly3(y_ext3, *popt3_3), y_ext3, 'r-', label=label3 % tuple(popt3_3))

if len(lines[3])!=0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4, 'b-', label='data')
    plt.plot(poly3(y_ext4, *popt4_3), y_ext4, 'r-', label=label3 % tuple(popt4_3))

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly3")
plt.xlabel('x')
plt.ylabel('y')

label4 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f,d=%5.3f, e=%5.3f'
plt.subplot(223)
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1, 'b-', label='data')
plt.plot(poly4(y_ext1, *popt1_4), y_ext1, 'r-', label=label4 % tuple(popt1_4))

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2, 'b-', label='data')
plt.plot(poly4(y_ext2, *popt2_4), y_ext2, 'r-', label=label4 % tuple(popt2_4))

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3, 'b-', label='data')
plt.plot(poly4(y_ext3, *popt3_4), y_ext3, 'r-', label=label4 % tuple(popt3_4))

if len(lines[3])!=0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4, 'b-', label='data')
    plt.plot(poly4(y_ext4, *popt4_4), y_ext4, 'r-', label=label4 % tuple(popt4_4))

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly4")
plt.xlabel('x')
plt.ylabel('y')


label5 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f,d=%5.3f, e=%5.3f, f=%5.3f'
plt.subplot(224)
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1, 'b-', label='data')
plt.plot(poly5(y_ext1, *popt1_5), y_ext1, 'r-', label=label5 % tuple(popt1_5))

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2, 'b-', label='data')
plt.plot(poly5(y_ext2, *popt2_5), y_ext2, 'r-', label=label5 % tuple(popt2_5))

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3, 'b-', label='data')
plt.plot(poly5(y_ext3, *popt3_5), y_ext3, 'r-', label=label5 % tuple(popt3_5))

if len(lines[3])!=0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4, 'b-', label='data')
    plt.plot(poly5(y_ext4, *popt4_5), y_ext4, 'r-', label=label5 % tuple(popt4_5))

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly5")
plt.xlabel('x')
plt.ylabel('y')

path_to_file_with_plots_name = './results_smoothed_lanes/frame' + str(FRAME_NUMBER) + '.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()
