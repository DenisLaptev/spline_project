import os
import re  # for reg exp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# popt:array=Optimal values for the parameters so that the sum
# of the squared residuals of f(xdata, *popt) - ydata is minimized

# pcov:2d array=The estimated covariance of popt.
# The diagonals provide the variance of the parameter estimate.


path = '../initial_images/'

# number of points for approximation
# N = 360
N = 1000

# height of image
HEIGHT = 360

# width of image
WIDTH = 640

S_TOTAL = 360 * 640

S_PERCENT_LIMIT = 0.6


# ------------POLINOMIALS_FUNCTIONS----------------

def f3(y, a, b, c, d):
    return a + b * y + c * y ** 2 + d * y ** 3


def f4(y, a, b, c, d, e):
    return a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4

def poly0(y_list, a):
    x_list = []
    for y in y_list:
        x_list.append(a)

    return x_list


def poly1(y_list, a, b):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y )

    return x_list


def poly2(y_list, a, b, c):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2)

    return x_list


def poly3(y_list, a, b, c, d):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3)

    return x_list


def poly4(y_list, a, b, c, d, e):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4)

    return x_list


def poly5(y_list, a, b, c, d, e, f):
    x_list = []
    for y in y_list:
        x_list.append(a + b * y + c * y ** 2 + d * y ** 3 + e * y ** 4 + f * y ** 5)

    return x_list


# ------------APPROXIMATION_FUNCTIONS----------------

def make_extended_coordinates_list(initial_coordinates_list, number_of_points):
    extended_coordinates_list = []
    ymin = np.min(initial_coordinates_list)
    ymax = np.max(initial_coordinates_list)
    delta = (ymax - ymin) / number_of_points
    for i in range(number_of_points):
        extended_coordinates_list.append(ymin + delta * i)
    return extended_coordinates_list


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


def get_xs_ys_lists_from_curve(popt, y1, y2, poly_order='poly4'):
    xs = []
    ys = []

    if len(popt)==2:
        a = popt[0]
        b = popt[1]
        c = 0
        d = 0
        e = 0
    elif len(popt)==3:
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = 0
        e = 0

    elif len(popt)==4:
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
            e = 0
    else:
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



    # print('y1=', y1)
    # print('y2=', y2)
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


def calculate_area_between_curves(popt, yps, xs, poly_order='poly4'):
    Sdelta = 0
    Sdelta_list = []
    yresult_list = []
    xcurve_result_list = []

    Ny = len(yps) - 1
    for j in range(Ny):
        y1 = yps[j]
        y2 = yps[j + 1]
        x1 = xs[j]
        x2 = xs[j + 1]
        # print("y1,y2=", str(y1), ",", str(y2))
        # print("len(y1),len(y2)=", len(str(y1)), ",", len(str(y2)))
        xs_straight, ys_straight = get_xs_ys_lists_from_straight(x1, x2, y1, y2)
        xs_curve, ys_curve = get_xs_ys_lists_from_curve(popt, y1, y2, poly_order)

        if j == Ny - 1:
            xs_straight.append(xs[Ny]), ys_straight.append(yps[Ny])
            if len(popt) == 2:
                a = popt[0]
                b = popt[1]
                c = 0
                d = 0
                e = 0
                x = f4(yps[Ny], a, b, c, d, e)
            elif len(popt) == 3:
                a = popt[0]
                b = popt[1]
                c = popt[2]
                d = 0
                e = 0
                x = f4(yps[Ny], a, b, c, d, e)

            elif len(popt) == 4:
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
                    e = 0
                    x = f4(yps[Ny], a, b, c, d, e)
            else:
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

        xcurve_result_list.extend(xs_curve)
        number_of_points = len(ys_straight)

        for i in range(number_of_points):
            delta = abs(xs_straight[i] - xs_curve[i])
            Sdelta_list.append(delta)
            yresult_list.append(ys_straight[i])
            Sdelta += delta
    #     print("-->ys_straight=", ys_straight)
    #     print("-->xs_straight=", xs_straight)
    #     print("-->ys_curve=", ys_curve)
    #     print("-->xs_curve=", xs_curve)
    # print("Sdelta=", Sdelta)
    # print("yresult_list=", yresult_list)
    # print("xcurve_result_list=", xcurve_result_list)
    # print("Sdelta_list=", Sdelta_list)

    x_int_list = []
    for x in xcurve_result_list:
        x_rounded=(round(x, 2))
        x_int_list.append(int(x_rounded))

    # print("len(yresult_list)=", len(yresult_list))
    # print("len(x_rounded)=", len(x_rounded))
    return Sdelta, yresult_list, x_int_list, Sdelta_list


# ------------DATA_EXTRACTION----------------
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


def get_lines_list_from_file(path_to_file):
    file = open(path_to_file, "r")
    lines = file.read().split("\n")
    not_empty_lines = []
    for line in lines:
        line.strip()

        if len(line) > 0:
            not_empty_lines.append(line)
    file.close()
    print("not_empty_lines=",not_empty_lines)
    return not_empty_lines


def get_xy_list_from_line(line):
    numbers_list = []
    elements = line.split(" ")
    for element in elements:
        if len(element) != 0:
            numbers_list.append(int(element))
    return numbers_list


def get_xs_ys_lists_from_xy_list(xy_list):
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


def get_xy_list_from_xs_ys_lists(xs_list, yps_list):
    # print("xy_list=", xy_list)
    xy_list = []

    ys = ys_matplotlib_to_ys_opencv(ys_matplotlib=yps_list, height=HEIGHT)
    xs = xs_list

    number_of_elements = len(xs_list)

    for i in range(number_of_elements):
        xy_list.append(xs[i])
        xy_list.append(ys[i])

    #print("xy_list=", xy_list)

    return xy_list


def get_list_of_frame_dicts(frame_numbers, file_names, paths_to_files):
    list_of_frame_dicts = []
    number_of_elements = len(frame_numbers)
    for i in range(number_of_elements):
        frame_dict = {}
        frame_dict['frame_number'] = frame_numbers[i]
        frame_dict['file_name'] = file_names[i]
        frame_dict['path_to_file'] = paths_to_files[i]
        lines = get_lines_list_from_file(path_to_file=paths_to_files[i])
        number_of_lines = len(lines)
        frame_dict['number_of_lines'] = number_of_lines
        lines_dict = {}
        for j in range(number_of_lines):
            key = 'line' + str(j + 1)
            lines_dict[key] = get_xy_list_from_line(lines[j])
        frame_dict['lines_dict'] = lines_dict
        #print("frame_dict=", frame_dict)
        list_of_frame_dicts.append(frame_dict)

    return list_of_frame_dicts


def leave_only_positive_xs_and_ys(all_xs, all_ys):
    positive_xs = []
    positive_ys = []

    number_of_elements = len(all_xs)
    for i in range(number_of_elements):
        if all_xs[i] >= 0:
            positive_xs.append(all_xs[i])
            positive_ys.append(all_ys[i])

    return positive_xs, positive_ys


def smoth_xy_list_line(initial_xy_list_line):
    smoothed_xy_list_line = []

    xs, yps = get_xs_ys_lists_from_xy_list(xy_list=initial_xy_list_line)
    # print('xs=', xs)
    # print('yps=', yps)

    if len(yps)<=1:
        # Fit for the parameters a, b, c, d, e of the function poly4
        popt, pcov = curve_fit(f=poly0,
                               xdata=yps,
                               ydata=xs)
    elif len(yps)==2:
        # Fit for the parameters a, b, c, d, e of the function poly4
        popt, pcov = curve_fit(f=poly1,
                               xdata=yps,
                               ydata=xs)

    elif len(yps)==3:
        # Fit for the parameters a, b, c, d, e of the function poly4
        popt, pcov = curve_fit(f=poly2,
                               xdata=yps,
                               ydata=xs)
    elif len(yps)==4:
        # Fit for the parameters a, b, c, d, e of the function poly4
        popt, pcov = curve_fit(f=poly3,
                               xdata=yps,
                               ydata=xs)
    else:
        popt, pcov = curve_fit(f=poly4,
                               xdata=yps,
                               ydata=xs)
    # print('popt=', popt)
    # print('pcov=', pcov)

    Sdelta, yresult_list, xcurve_result_list, Sdelta_list = calculate_area_between_curves(
        popt=popt,
        yps=yps,
        xs=xs, poly_order='poly4')
    # print("yresult_list=", yresult_list)
    # print("xcurve_result_list=", xcurve_result_list)
    # print('Sdelta_list=', Sdelta_list)

    S_error = Sdelta / S_TOTAL * 100

    if S_error < S_PERCENT_LIMIT:
        # print("S_error=", S_error)
        # print("len(yresult_list)=", len(yresult_list))
        # print("len(xcurve_result_list)=", len(xcurve_result_list))

        positive_xs, positive_ys = leave_only_positive_xs_and_ys(all_xs=xcurve_result_list,
                                                                 all_ys=yresult_list)

        xy_list = get_xy_list_from_xs_ys_lists(xs_list=positive_xs,
                                               yps_list=positive_ys)
        smoothed_xy_list_line = xy_list
    else:
        smoothed_xy_list_line = initial_xy_list_line

    return smoothed_xy_list_line


def smooth_transform_of_frame_dict(frame_dict):
    # change:
    # frame_dict['file_name']
    # frame_dict['path_to_file']
    # frame_dict['lines_dict']
    #
    smoothed_frame_dict = {}

    initial_frame_number = frame_dict['frame_number']
    initial_file_name = frame_dict['file_name']
    initial_path_to_file = frame_dict['path_to_file']
    initial_number_of_lines = frame_dict['number_of_lines']
    initial_lines_dict = frame_dict['lines_dict']

    print('FRAME NUMBER=',initial_frame_number)

    smoothed_lines_dict = {}
    if len(initial_lines_dict)!=0:

        for j in range(initial_number_of_lines):
            key = 'line' + str(j + 1)

            initial_xy_list_line = initial_lines_dict[key]
            print('xy_list_line=', initial_xy_list_line)

            smoothed_xy_list_line = []
            smoothed_xy_list_line = smoth_xy_list_line(initial_xy_list_line)
            print('smoothed_xy_list_line=', smoothed_xy_list_line)

            smoothed_lines_dict[key] = smoothed_xy_list_line

    modified_file_name = initial_file_name
    modified_path_to_file = '../Smoothed/' + modified_file_name

    smoothed_frame_dict['frame_number'] = initial_frame_number
    smoothed_frame_dict['file_name'] = modified_file_name
    smoothed_frame_dict['path_to_file'] = modified_path_to_file
    smoothed_frame_dict['number_of_lines'] = initial_number_of_lines
    smoothed_frame_dict['lines_dict'] = smoothed_lines_dict
    return smoothed_frame_dict


def get_list_of_smoothed_frame_dicts(list_of_frame_dicts):
    list_of_smoothed_frame_dicts=[]

    #print("len(list_of_frame_dicts)=",len(list_of_frame_dicts))

    for frame_dict in list_of_frame_dicts:
        # if len(frame_dict['lines_dict'])==0:
        #     print("ZERO!")
        smoothed_frame_dict = smooth_transform_of_frame_dict(frame_dict=frame_dict)
        list_of_smoothed_frame_dicts.append(smoothed_frame_dict)

    return list_of_smoothed_frame_dicts


def create_result_images(xs, ys, frame_number, line_number, poly_order='poly4'):
    image_title = poly_order + '_frame' + str(frame_number) + '_line' + str(line_number)
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
    path_to_result_image = './' + image_title + '.png'
    cv2.imwrite(path_to_result_image, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_lines_list_to_file(lines_list,path_to_file):

    with open(path_to_file, 'w') as f:
        for xy_list_line in lines_list:
            print('line_to_save=',xy_list_line)
            for element in xy_list_line:
                f.write("%s " % element)
            f.write("\n")

def generate_files_with_smoothed_curves(list_of_smoothed_frame_dicts):
    for frame_dict in list_of_smoothed_frame_dicts:
        number_of_lines = frame_dict['number_of_lines']
        lines_dict = frame_dict['lines_dict']
        lines_list = []
        for i in range(number_of_lines):
            key = 'line' + str(i + 1)
            line = lines_dict[key]
            lines_list.append(line)
        #print('lines_list=', lines_list)
        path_to_file = frame_dict['path_to_file']
        save_lines_list_to_file(lines_list, path_to_file)


def main():
    frame_numbers, file_names, paths_to_files = get_data_lists_from_folder(path_to_folder=path)
    # print('frame_numbers=', frame_numbers)
    # print('file_names=', file_names)
    # print('paths_to_files=', paths_to_files)
    list_of_frame_dicts = get_list_of_frame_dicts(frame_numbers=frame_numbers,
                                                  file_names=file_names,
                                                  paths_to_files=paths_to_files)
    print('=============INITIAL==============')
    #print('list_of_frame_dicts[7]=',list_of_frame_dicts[7])
    print('list_of_frame_dicts=',list_of_frame_dicts)

    #smoothed_frame_dict = smooth_transform_of_frame_dict(frame_dict=list_of_frame_dicts[7])

    print('=============SMOOTHED==============')
    #print('smoothed_frame_dict=', smoothed_frame_dict)


    list_of_smoothed_frame_dicts = get_list_of_smoothed_frame_dicts(list_of_frame_dicts=list_of_frame_dicts)
    print('list_of_smoothed_frame_dicts=', list_of_smoothed_frame_dicts)


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    generate_files_with_smoothed_curves(list_of_smoothed_frame_dicts)




if __name__ == "__main__":
    main()
