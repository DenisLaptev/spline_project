import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2

import sympy as sy
from scipy.integrate import quad

# popt:array=Optimal values for the parameters so that the sum
# of the squared residuals of f(xdata, *popt) - ydata is minimized

# pcov:2d array=The estimated covariance of popt.
# The diagonals provide the variance of the parameter estimate.

# Frames 239, 544, 594, 614, 877, 2305, 2316, 2616
FRAME_NUMBER = 2616
PATH_TO_INITIAL_FILE_Nagaraju = "../initial_images/_VL_Nagaraju_frame_" + str(FRAME_NUMBER) + ".lines.txt"
PATH_TO_INITIAL_FILE_ramarajuk = "../initial_images/_VL_ramarajuk_frame_" + str(FRAME_NUMBER) + ".lines.txt"

if FRAME_NUMBER in [239, 544, 594, 614, 877]:
    path_to_initial_file = PATH_TO_INITIAL_FILE_Nagaraju
elif FRAME_NUMBER in [2305, 2316, 2616]:
    path_to_initial_file = PATH_TO_INITIAL_FILE_ramarajuk

# number of points for approximation
# N = 360
N = 1000

# height of image
HEIGHT = 360

# width of image
WIDTH = 640


def f3(y, a, b, c, d):
    return a + b * y + c * y ** 2 + d * y ** 3


def f4(y, a, b, c, d, e):
    return a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4


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


def get_lines_list_from_file(frame_number, path_to_file):
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


def calculate_curves_diff(yps, xs, x_poly):
    ys = []
    diffs = []

    number_of_elements = len(yps)
    for i in range(number_of_elements):
        ys.append(yps[i])
        diffs.append((x_poly[i] - xs[i]) ** 1)
        # print('yps[i]=', yps[i])
        # print('x_poly[i]=', x_poly[i])
        # print('xs[i]=', xs[i])
    print('ys_for_diff=', ys)
    print('diffs=', diffs)

    return ys, diffs


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


def calculate_area_trap(x1, x2, y1, y2):
    Str = (x1 + x2) / 2 * (y2 - y1)
    return Str


def calculate_area_integral(poly_order, popt, y1, y2):
    Sint = 0

    if poly_order == 'poly3':
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        Sint, error = quad(func=f3, a=y1, b=y2, args=(a, b, c, d))
    elif poly_order == 'poly4':
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        e = popt[4]
        Sint, error = quad(func=f4, a=y1, b=y2, args=(a, b, c, d, e))

    return Sint


def calculate_area_between_curves(poly_order, popt,  yps,xs):
    Sdelta=0
    Sdelta_list=[]
    yresult_list=[]

    Ny=len(yps)-1
    for j in range(Ny):
        y1=yps[j]
        y2=yps[j+1]
        x1 = xs[j]
        x2 = xs[j + 1]
        print("y1,y2=",str(y1),",",str(y2))
        xs_straight,ys_straight=get_xs_ys_lists_from_straight(x1, x2, y1, y2)
        xs_curve,ys_curve=get_xs_ys_lists_from_curve(poly_order, popt, y1, y2)
        if j==Ny-1:
            xs_straight.append(xs[Ny]), ys_straight.append(yps[Ny])
            if poly_order == 'poly3':
                a = popt[0]
                b = popt[1]
                c = popt[2]
                d = popt[3]
                x = f3(yps[Ny], a, b, c, d)
            elif poly_order == 'poly4':
                a = popt[0]
                b = popt[1]
                c = popt[2]
                d = popt[3]
                e = popt[4]
                x = f4(yps[Ny], a, b, c, d, e)

            xs_curve.append(x), ys_curve.append(yps[Ny])
        number_of_points=len(ys_straight)


        for i in range(number_of_points):
            delta=abs(xs_straight[i]-xs_curve[i])
            Sdelta_list.append(delta)
            yresult_list.append(ys_straight[i])
            Sdelta+=delta
        print("-->ys_straight=", ys_straight)
        print("-->xs_straight=", xs_straight)
        print("-->ys_curve=", ys_curve)
        print("-->xs_curve=", xs_curve)
        # if j==Ny-1:
        #     print("-->ys_straight=",ys_straight)
    print("Sdelta=",Sdelta)
    print("yresult_list=",yresult_list)
    print("Sdelta_list=",Sdelta_list)
    return Sdelta, yresult_list, Sdelta_list


def create_result_images(xs, ys, method_name, frame_number, line_number):
    image_title = method_name + '_frame' + str(frame_number) + '_line' + str(line_number)
    height = HEIGHT
    width = WIDTH
    image = np.zeros((height, width), np.uint8)
    points_number = len(xs)
    for i in range(points_number):
        x = int(xs[i])
        y = int(ys[i])
        # print('x=',x)
        # print('y=',y)
        if x >= 0 and y >= 0:
            image[y, x] = 255
    cv2.imshow("image_title", image)
    path_to_result_image = './results_metrix/' + image_title + '.png'
    cv2.imwrite(path_to_result_image, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


lines = get_lines_list_from_file(frame_number=FRAME_NUMBER,
                                 path_to_file=path_to_initial_file)
xs1, yps1 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[0]))
xs2, yps2 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[1]))
xs3, yps3 = generate_xs_ys_lists_from_xy_list(xy_list=get_numbers_list_from_line(lines[2]))
if len(lines[3]) != 0:
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
if len(lines[3]) != 0:
    popt4_2, pcov4_2 = curve_fit(f=poly2,
                                 xdata=yps4,
                                 ydata=xs4)
# print("popt1_2=", popt1_2)
# print("pcov1_2=", pcov1_2)

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
if len(lines[3]) != 0:
    popt4_3, pcov4_3 = curve_fit(f=poly3,
                                 xdata=yps4,
                                 ydata=xs4)

# print("popt1_3=", popt1_3)
# print("pcov1_3=", pcov1_3)

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
if len(lines[3]) != 0:
    popt4_4, pcov4_4 = curve_fit(f=poly4,
                                 xdata=yps4,
                                 ydata=xs4)
# print("popt1_4=", popt1_4)
# print("pcov1_4=", pcov1_4)

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
if len(lines[3]) != 0:
    popt4_5, pcov4_5 = curve_fit(f=poly5,
                                 xdata=yps4,
                                 ydata=xs4)
# print("popt1_5=", popt1_5)
# print("pcov1_5=", pcov1_5)


# ---------------------------------initial_figure----------------------------
title_of_figure = "FRAME_" + str(FRAME_NUMBER)

#plt.figure(figsize=(20, 10))
plt.suptitle(title_of_figure)
plt.plot(xs1, yps1)
plt.plot(xs2, yps2)
plt.plot(xs3, yps3)
if len(lines[3]) != 0:
    plt.plot(xs4, yps4)
plt.xlim((0, 640))
plt.ylim((0, 360))
plt.xlabel('x')
plt.ylabel('y')
path_to_file_with_plots_name = './results_metrix/initial_frame' + str(FRAME_NUMBER) + '.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()

# ---------------------------------result_figure_3----------------------------
title_of_figure = "FRAME_" + str(FRAME_NUMBER)
#plt.figure(figsize=(20, 10))
plt.suptitle(title_of_figure)


label3 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f,d=%5.3f'
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1,
         'b-',
         label='data')
plt.plot(poly3(y_ext1, *popt1_3), y_ext1,
         'r-',
         label=label3 % tuple(popt1_3))

ys_metr1_3, diffs_metr1_3 = calculate_curves_diff(yps=yps1, xs=xs1, x_poly=poly3(yps1, *popt1_3))
create_result_images(xs=poly3(y_ext1, *popt1_3),
                     ys=ys_matplotlib_to_ys_opencv(y_ext1, height=HEIGHT),
                     method_name='poly3',
                     frame_number=FRAME_NUMBER,
                     line_number=1)
Sdelta1_3, yps1_3, Sdelta_list1_3 = calculate_area_between_curves(poly_order='poly3',
                                                                  popt=popt1_3,
                                                                  yps=yps1,
                                                                  xs=xs1)
#print("Obtained Sdelta1_3=", Sdelta1_3)

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2,
         'b-',
         label='data')
plt.plot(poly3(y_ext2, *popt2_3), y_ext2,
         'r-',
         label=label3 % tuple(popt2_3))
ys_metr2_3, diffs_metr2_3 = calculate_curves_diff(yps=yps2, xs=xs2, x_poly=poly3(yps2, *popt2_3))
create_result_images(xs=poly3(y_ext2, *popt2_3),
                     ys=ys_matplotlib_to_ys_opencv(y_ext2, height=HEIGHT),
                     method_name='poly3',
                     frame_number=FRAME_NUMBER,
                     line_number=2)
Sdelta2_3, yps2_3, Sdelta_list2_3 = calculate_area_between_curves(poly_order='poly3',
                                                                  popt=popt2_3,
                                                                  yps=yps2,
                                                                  xs=xs2)
#print("Obtained Sdelta2_3=", Sdelta2_3)

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3,
         'b-',
         label='data')
plt.plot(poly3(y_ext3, *popt3_3), y_ext3,
         'r-',
         label=label3 % tuple(popt3_3))
ys_metr3_3, diffs_metr3_3 = calculate_curves_diff(yps=yps3, xs=xs3, x_poly=poly3(yps3, *popt3_3))
create_result_images(xs=poly3(y_ext3, *popt3_3),
                     ys=ys_matplotlib_to_ys_opencv(y_ext3, height=HEIGHT),
                     method_name='poly3',
                     frame_number=FRAME_NUMBER,
                     line_number=3)
Sdelta3_3, yps3_3, Sdelta_list3_3 = calculate_area_between_curves(poly_order='poly3',
                                                                  popt=popt3_3,
                                                                  yps=yps3,
                                                                  xs=xs3)
#print("Obtained Sdelta3_3=", Sdelta3_3)

if len(lines[3]) != 0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4,
             'b-',
             label='data')
    plt.plot(poly3(y_ext4, *popt4_3), y_ext4,
             'r-',
             label=label3 % tuple(popt4_3))
    ys_metr4_3, diffs_metr4_3 = calculate_curves_diff(yps=yps4, xs=xs4, x_poly=poly3(yps4, *popt4_3))
    create_result_images(xs=poly3(y_ext4, *popt4_3),
                         ys=ys_matplotlib_to_ys_opencv(y_ext4, height=HEIGHT),
                         method_name='poly3',
                         frame_number=FRAME_NUMBER,
                         line_number=4)
    Sdelta4_3, yps4_3, Sdelta_list4_3 = calculate_area_between_curves(poly_order='poly3',
                                                                      popt=popt4_3,
                                                                      yps=yps4,
                                                                      xs=xs4)
    #print("Obtained Sdelta4_3=", Sdelta4_3)

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly3")
plt.xlabel('x')
plt.ylabel('y')

path_to_file_with_plots_name = './results_metrix/frame' + str(FRAME_NUMBER) + '_poly3.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()

# ---------------------------------result_figure_4----------------------------
title_of_figure = "FRAME_" + str(FRAME_NUMBER)
#plt.figure(figsize=(20, 10))
plt.suptitle(title_of_figure)

label4 = 'fit: a=%5.3f, b=%5.3f, c=%5.3f,d=%5.3f, e=%5.3f'
# y_extended
y_ext1 = make_extended_coordinates_list(initial_coordinates_list=yps1, number_of_points=N)
plt.plot(xs1, yps1,
         'b-',
         label='data')
plt.plot(poly4(y_ext1, *popt1_4), y_ext1,
         'r-',
         label=label4 % tuple(popt1_4))
ys_metr1_4, diffs_metr1_4 = calculate_curves_diff(yps=yps1, xs=xs1, x_poly=poly4(yps1, *popt1_4))
create_result_images(xs=poly4(y_ext1, *popt1_4),
                     ys=ys_matplotlib_to_ys_opencv(y_ext1, height=HEIGHT),
                     method_name='poly4',
                     frame_number=FRAME_NUMBER,
                     line_number=1)
Sdelta1_4, yps1_4, Sdelta_list1_4 = calculate_area_between_curves(poly_order='poly4',
                                                                  popt=popt1_4,
                                                                  yps=yps1,
                                                                  xs=xs1)
#print("Obtained Sdelta1_4=", Sdelta1_4)

# y_extended
y_ext2 = make_extended_coordinates_list(initial_coordinates_list=yps2, number_of_points=N)
plt.plot(xs2, yps2,
         'b-',
         label='data')
plt.plot(poly4(y_ext2, *popt2_4), y_ext2,
         'r-',
         label=label4 % tuple(popt2_4))
ys_metr2_4, diffs_metr2_4 = calculate_curves_diff(yps=yps2, xs=xs2, x_poly=poly4(yps2, *popt2_4))
create_result_images(xs=poly4(y_ext2, *popt2_4),
                     ys=ys_matplotlib_to_ys_opencv(y_ext2, height=HEIGHT),
                     method_name='poly4',
                     frame_number=FRAME_NUMBER,
                     line_number=2)
Sdelta2_4, yps2_4, Sdelta_list2_4 = calculate_area_between_curves(poly_order='poly4',
                                                                  popt=popt2_4,
                                                                  yps=yps2,
                                                                  xs=xs2)
#print("Obtained Sdelta2_4=", Sdelta2_4)

# y_extended
y_ext3 = make_extended_coordinates_list(initial_coordinates_list=yps3, number_of_points=N)
plt.plot(xs3, yps3, 'b-', label='data')
plt.plot(poly4(y_ext3, *popt3_4), y_ext3,
         'r-',
         label=label4 % tuple(popt3_4))
ys_metr3_4, diffs_metr3_4 = calculate_curves_diff(yps=yps3, xs=xs3, x_poly=poly4(yps3, *popt3_4))
create_result_images(xs=poly4(y_ext3, *popt3_4),
                     ys=ys_matplotlib_to_ys_opencv(y_ext3, height=HEIGHT),
                     method_name='poly4',
                     frame_number=FRAME_NUMBER,
                     line_number=3)
Sdelta3_4, yps3_4, Sdelta_list3_4 = calculate_area_between_curves(poly_order='poly4',
                                                                  popt=popt3_4,
                                                                  yps=yps3,
                                                                  xs=xs3)
#print("Obtained Sdelta3_4=", Sdelta3_4)

if len(lines[3]) != 0:
    # y_extended
    y_ext4 = make_extended_coordinates_list(initial_coordinates_list=yps4, number_of_points=N)
    plt.plot(xs4, yps4, 'b-', label='data')
    plt.plot(poly4(y_ext4, *popt4_4), y_ext4,
             'r-',
             label=label4 % tuple(popt4_4))
    ys_metr4_4, diffs_metr4_4 = calculate_curves_diff(yps=yps4, xs=xs4, x_poly=poly4(yps4, *popt4_4))
    create_result_images(xs=poly4(y_ext4, *popt4_4),
                         ys=ys_matplotlib_to_ys_opencv(y_ext4, height=HEIGHT),
                         method_name='poly4',
                         frame_number=FRAME_NUMBER,
                         line_number=4)
    Sdelta4_4, yps4_4, Sdelta_list4_4 = calculate_area_between_curves(poly_order='poly4',
                                                                      popt=popt4_4,
                                                                      yps=yps4,
                                                                      xs=xs4)
    print("Obtained Sdelta4_4=", Sdelta4_4)

plt.xlim((0, 640))
plt.ylim((0, 360))
plt.title("poly4")
plt.xlabel('x')
plt.ylabel('y')

path_to_file_with_plots_name = './results_metrix/frame' + str(FRAME_NUMBER) + '_poly4.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()


title_of_figure = "metrics_diff1_FRAME_"+str(FRAME_NUMBER)
plt.figure(figsize=(20, 10))
plt.suptitle(title_of_figure)

plt.subplot(221)
plt.plot(ys_metr1_3, diffs_metr1_3, label='poly_3')
plt.plot(ys_metr1_4, diffs_metr1_4, label='poly_4')
plt.title("line1")
plt.xlabel('y')
plt.ylabel('diff1,px')
plt.legend()
plt.subplot(222)
plt.plot(ys_metr2_3, diffs_metr2_3, label='poly_3')
plt.plot(ys_metr2_4, diffs_metr2_4, label='poly_4')
plt.title("line2")
plt.xlabel('y')
plt.ylabel('diff1,px')
plt.legend()
plt.subplot(223)
plt.plot(ys_metr3_3, diffs_metr3_3, label='poly_3')
plt.plot(ys_metr3_4, diffs_metr3_4, label='poly_4')
plt.title("line3")
plt.xlabel('y')
plt.ylabel('diff1,px')
plt.legend()
if len(lines[3]) != 0:
    plt.subplot(224)
    plt.plot(ys_metr4_3, diffs_metr4_3, label='poly_3')
    plt.plot(ys_metr4_4, diffs_metr4_4, label='poly_4')
    plt.title("line4")
    plt.xlabel('y')
    plt.ylabel('diff1,px')
    plt.legend()
path_to_file_with_plots_name = './results_metrix/frame' + str(FRAME_NUMBER) + '_metrics_diff1.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()

title_of_figure = "metrics_area_between_FRAME_"+str(FRAME_NUMBER)
plt.figure(figsize=(20, 10))
plt.suptitle(title_of_figure)


Sdelta1_3=round(Sdelta1_3/640/360*100,5)
Sdelta1_4=round(Sdelta1_4/640/360*100,5)
Sdelta2_3=round(Sdelta2_3/640/360*100,5)
Sdelta2_4=round(Sdelta2_4/640/360*100,5)
Sdelta3_3=round(Sdelta3_3/640/360*100,5)
Sdelta3_4=round(Sdelta3_4/640/360*100,5)
if len(lines[3]) != 0:
    Sdelta4_3=round(Sdelta4_3/640/360*100,5)
    Sdelta4_4=round(Sdelta4_4/640/360*100,5)

plt.subplot(221)
plt.plot(yps1_3, Sdelta_list1_3, label='poly3_dS=' + str(Sdelta1_3)+'%')
plt.plot(yps1_4, Sdelta_list1_4, label='poly4_dS=' + str(Sdelta1_4)+'%')
plt.title("line1")
plt.xlabel('y')
plt.ylabel('area_diff,px')
plt.legend()
plt.subplot(222)
plt.plot(yps2_3, Sdelta_list2_3, label='poly3_dS=' + str(Sdelta2_3)+'%')
plt.plot(yps2_4, Sdelta_list2_4, label='poly4_dS=' + str(Sdelta2_4)+'%')
plt.title("line2")
plt.xlabel('y')
plt.ylabel('area_diff,px')
plt.legend()
plt.subplot(223)
plt.plot(yps3_3, Sdelta_list3_3, label='poly3_dS=' + str(Sdelta3_3)+'%')
plt.plot(yps3_4, Sdelta_list3_4, label='poly4_dS=' + str(Sdelta3_4)+'%')
plt.title("line3")
plt.xlabel('y')
plt.ylabel('area_diff,px')
plt.legend()
if len(lines[3]) != 0:
    plt.subplot(224)
    plt.plot(yps4_3, Sdelta_list4_3, label='poly3_dS=' + str(Sdelta4_3)+'%')
    plt.plot(yps4_4, Sdelta_list4_4, label='poly4_dS=' + str(Sdelta4_4)+'%')
    plt.title("line4")
    plt.xlabel('y')
    plt.ylabel('area_diff,px')
    plt.legend()
path_to_file_with_plots_name = './results_metrix/frame' + str(FRAME_NUMBER) + '_metrics_area_between.png'
plt.savefig(path_to_file_with_plots_name)
plt.show()


