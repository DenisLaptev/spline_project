import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#popt:array=Optimal values for the parameters so that the sum
# of the squared residuals of f(xdata, *popt) - ydata is minimized

# pcov:2d array=The estimated covariance of popt.
# The diagonals provide the variance of the parameter estimate.


line1=[0, 279,
       25, 259,
       120, 239,
       159, 219,
       294, 199,
       395, 179]


line2=[82, 359,
       105, 339,
       134, 319,
       168, 299,
       202, 279,
       237, 259,
       275, 239,
       318, 219,
       392, 199,
       454, 179]

line3=[549, 359,
       537, 339,
       522, 319,
       509, 299,
       492, 279,
       468, 259,
       456, 239,
       442, 219,
       443, 199,
       489, 179]

length1= len(line1)
length2= len(line2)
length3= len(line3)

xs1=[]
ys1=[]
yps1=[]
for i in range(length1):
    if i%2==0:
        xs1.append(line1[i])
    else:
        ys1.append(line1[i])

for i in range(len(ys1)):
    yps1.append(360-ys1[i])

print("xs1=",xs1)
print("ys1=",ys1)
print("yps1=",yps1)



xs2=[]
ys2=[]
yps2=[]
for i in range(length2):
    if i%2==0:
        xs2.append(line2[i])
    else:
        ys2.append(line2[i])

for i in range(len(ys2)):
    yps2.append(360-ys2[i])

print("xs2=",xs2)
print("ys2=",ys2)
print("yps2=",yps2)



xs3=[]
ys3=[]
yps3=[]
for i in range(length3):
    if i%2==0:
        xs3.append(line3[i])
    else:
        ys3.append(line3[i])

for i in range(len(ys3)):
    yps3.append(360-ys3[i])

print("xs3=",xs3)
print("ys3=",ys3)
print("yps3=",yps3)

#plt.figure(num=None, figsize=(360, 640), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(121)
plt.plot(xs1,ys1)
plt.plot(xs2,ys2)
plt.plot(xs3,ys3)


plt.subplot(122)
plt.plot(xs1,yps1)
plt.plot(xs2,yps2)
plt.plot(xs3,yps3)
plt.show()


plt.plot(xs1,yps1)
plt.plot(xs2,yps2)
plt.plot(xs3,yps3)
plt.xlim((0, 640))
plt.ylim((0, 360))
plt.show()
