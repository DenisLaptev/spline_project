import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# popt:array=Optimal values for the parameters so that the sum
# of the squared residuals of f(xdata, *popt) - ydata is minimized

# pcov:2d array=The estimated covariance of popt.
# The diagonals provide the variance of the parameter estimate.


# Define the data to be fit with some noise
xdata = np.linspace(start=0, stop=4, num=50)
y = func(x=xdata, a=2.5, b=1.3, c=0.5)
print("y=",y)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise

plt.plot(xdata, ydata, 'b-', label='data')

# Fit for the parameters a, b, c of the function func
popt, pcov = curve_fit(f=func,
                       xdata=xdata,
                       ydata=ydata)
print("popt=", popt)
print("pcov=", pcov)

plt.plot(xdata,
         func(xdata, *popt),
         'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(f=func,
                       xdata=xdata,
                       ydata=ydata,
                       bounds=(0, [3., 1., 0.5]))
print("popt=", popt)
# print("pcov=", pcov)

plt.plot(xdata,
         func(xdata, *popt),
         'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
