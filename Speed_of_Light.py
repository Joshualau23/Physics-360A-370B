# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:27:49 2018

@author: Josh
"""

#Trial 1

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

Trial1x = [48,103,202,299,401,501,605,703,802,903,1000,1085,1166]
Trial1y = [2.6478,2.648,2.6586,2.6802,2.7103,2.7533,2.7973,2.8366,2.8715,2.9009,2.9342,2.9703,2.9985]
error1x = [1,1,3,3,1,1,1,2,2,1,1,1,1]
error1y = [0.0003,0.0005,0.0003,0.0003,.0005,0.001,0.002,0.001,.0005,0.001,0.002,0.001,0.001]

Trial1xup = []
Trial1yup = []
Trial1xdown = []
Trial1ydown = []

for i in range(0,len(Trial1x)):
    a = Trial1x[i] + error1x[i]
    b = Trial1y[i] + error1y[i]
    c = Trial1x[i] - error1x[i]
    d = Trial1y[i] - error1y[i]
    Trial1xup.append(a)
    Trial1yup.append(b)
    Trial1xdown.append(c)
    Trial1ydown.append(d)



b, m = polyfit(Trial1x, Trial1y, 1)
bup,mup = polyfit(Trial1xup,Trial1yup,1)
bdown,mdown = polyfit(Trial1xdown,Trial1ydown,1)



slope_error = ((mup-mdown)/2) 
intercept_error = ((bup-bdown)/2)

print b,m
print bup,mup
print bdown,mdown

print intercept_error, slope_error


x = np.arange(1200)


plt.errorbar(Trial1x,Trial1y,xerr=error1x,yerr=error1y,fmt = '+',color='r')
plt.scatter(Trial1x,Trial1y,marker=".",color = 'k')
plt.plot(x, b + m * x, '-',color = 'y')
plt.title('Trial 1 Linearized')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.6,ymax=3.10)
plt.savefig('PATrial1.pdf')
plt.show()

plt.errorbar(Trial1x,Trial1y,xerr=error1x,yerr=error1y,fmt = '+',color='r')
plt.scatter(Trial1x,Trial1y,marker=".",color = 'k')
plt.title('Trial 1')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.6,ymax=3.10)
plt.savefig('PATrial1raw.pdf')

#Trial 2

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

Trial2x = [36,100,197,304,402,503,604,698,805,902,1004,1103]
Trial2y = [2.6713,2.6723,2.6756,2.6870,2.7083,2.7592,2.7971,2.8342,2.8705,2.8650,2.9359,2.9743]
error2x = [1,2,1,2,2,2,1,1,1,2,1,1]
error2y = [0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007]

Trial2xup = []
Trial2yup = []
Trial2xdown = []
Trial2ydown = []

for i in range(0,len(Trial2x)):
    a = Trial2x[i] + error2x[i]
    b = Trial2y[i] + error2y[i]
    c = Trial2x[i] - error2x[i]
    d = Trial2y[i] - error2y[i]
    Trial2xup.append(a)
    Trial2yup.append(b)
    Trial2xdown.append(c)
    Trial2ydown.append(d)



b, m = polyfit(Trial2x, Trial2y, 1)
bup,mup = polyfit(Trial2xup,Trial2yup,1)
bdown,mdown = polyfit(Trial2xdown,Trial2ydown,1)


slope_error = ((mup-mdown)/2) 
intercept_error = ((bup-bdown)/2)

print b,m
print bup,mup
print bdown,mdown
print intercept_error, slope_error
x = np.arange(1200)

b2, m2 = polyfit(Trial2x, Trial2y, 1)

plt.errorbar(Trial2x,Trial2y,xerr=error2x,yerr=error2y,fmt = '+',color='r')
plt.scatter(Trial2x,Trial2y,marker=".",color = 'k')
plt.plot(x, b2 + m2 * x, '-',color = 'y')
plt.title('Trial 2 Linearized')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.yticks(np.arange(2.5, 3.05, step=0.05))
plt.ylim(ymin=2.6,ymax=3.05)
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.savefig('PATrial2.pdf')
plt.show()


plt.errorbar(Trial2x,Trial2y,xerr=error2x,yerr=error2y,fmt = '+',color='r')
plt.scatter(Trial2x,Trial2y,marker=".",color = 'k')
plt.title('Trial 2')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.yticks(np.arange(2.5, 3.05, step=0.05))
plt.ylim(ymin=2.6,ymax=3.05)
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.savefig('PATrial2raw.pdf')



#Trial 3

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

Trial3x = [107,198,299,400,497,597,700,802,900,1000,1094]
Trial3y = [2.6171,2.6649,2.6889,2.7225,2.7581,2.7929,2.8463,2.8789,2.9068,2.9941,2.9763]
error3x = [2,1,2,1,1,1,1,1,1,1,1]
error3y = [0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007]

Trial3xup = []
Trial3yup = []
Trial3xdown = []
Trial3ydown = []

for i in range(0,len(Trial3x)):
    a = Trial3x[i] + error3x[i]
    b = Trial3y[i] + error3y[i]
    c = Trial3x[i] - error3x[i]
    d = Trial3y[i] - error3y[i]
    Trial3xup.append(a)
    Trial3yup.append(b)
    Trial3xdown.append(c)
    Trial3ydown.append(d)



b, m = polyfit(Trial3x, Trial3y, 1)
bup,mup = polyfit(Trial3xup,Trial3yup,1)
bdown,mdown = polyfit(Trial3xdown,Trial3ydown,1)


slope_error = ((mup-mdown)/2) 
intercept_error = ((bup-bdown)/2)

print b,m
print bup,mup
print bdown,mdown
print intercept_error, slope_error

b3, m3 = polyfit(Trial3x, Trial3y, 1)

x = np.arange(1200)

plt.errorbar(Trial3x,Trial3y,xerr=error3x,yerr=error3y,fmt = '+',color='r')
plt.scatter(Trial3x,Trial3y,marker=".",color = 'k')
plt.plot(x, b3 + m3 * x, '-',color = 'y')
plt.title('Trial 3 Linearized')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.5,ymax=3.10)
plt.savefig('PATrial3.pdf')
plt.show()


plt.errorbar(Trial3x,Trial3y,xerr=error3x,yerr=error3y,fmt = '+',color='r')
plt.scatter(Trial3x,Trial3y,marker=".",color = 'k')
plt.title('Trial 3')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.5,ymax=3.10)
plt.savefig('PATrial3raw.pdf')



#Graphing

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math

###

Trial1x = [48,103,202,299,401,501,605,703,802,903,1000,1085,1166]
Trial1y = [2.6478,2.648,2.6586,2.6802,2.7103,2.7533,2.7973,2.8366,2.8715,2.9009,2.9342,2.9703,2.9985]

error1x = [1,1,3,3,1,1,1,2,2,1,1,1,1]
error1y = [0.0003,0.0005,0.0003,0.0003,.0005,0.001,0.002,0.001,.0005,0.001,0.002,0.001,0.001]


b, m = polyfit(Trial1x, Trial1y, 1)



x = np.arange(1200)


plt.errorbar(Trial1x,Trial1y,xerr=error1x,yerr=error1y,fmt = '+',color='r')
plt.scatter(Trial1x,Trial1y,marker=".",color = 'k')
plt.plot(x, b + m * x, '-',color = 'y')
plt.title('Trial 1')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.6,ymax=3.10)
plt.show()

###

Trial2x = [36,100,197,304,402,503,604,698,805,902,1004,1103]
Trial2y = [2.6713,2.6723,2.6756,2.6870,2.7083,2.7592,2.7971,2.8342,2.8705,2.8650,2.9359,2.9743]
error2x = [1,2,1,2,2,2,1,1,1,2,1,1]
error2y = [0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007]



b2, m2 = polyfit(Trial2x, Trial2y, 1)

plt.errorbar(Trial2x,Trial2y,xerr=error2x,yerr=error2y,fmt = '+',color='r')
plt.scatter(Trial2x,Trial2y,marker=".",color = 'k')
plt.plot(x, b2 + m2 * x, '-',color = 'y')
plt.title('Trial 2')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.yticks(np.arange(2.5, 3.05, step=0.05))
plt.ylim(ymin=2.6,ymax=3.05)
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.show()




###


Trial3x = [107,198,299,400,497,597,700,802,900,1000,1094]
Trial3y = [2.6171,2.6649,2.6889,2.7225,2.7581,2.7929,2.8463,2.8789,2.9068,2.9941,2.9763]
error3x = [2,1,2,1,1,1,1,1,1,1,1]
error3y = [0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007,0.0007]




b3, m3 = polyfit(Trial3x, Trial3y, 1)

x = np.arange(1200)

plt.errorbar(Trial3x,Trial3y,xerr=error3x,yerr=error3y,fmt = '+',color='r')
plt.scatter(Trial3x,Trial3y,marker=".",color = 'k')
plt.plot(x, b3 + m3 * x, '-',color = 'y')
plt.title('Trial 3')
plt.xlabel('Hertz (Hz)')
plt.ylabel('Travelling Microscope Position (cm)')
plt.xticks(np.arange(0, 1200, step=100))
plt.xlim(xmin=0,xmax=1200)
plt.yticks(np.arange(2.5, 3.10, step=0.05))
plt.ylim(ymin=2.5,ymax=3.10)
plt.show()

print b3,m3
print b2,m2
print b,m

c = (m  +m3)/200

cl = 4*3.14159*(20-7.108)*(7.108) / c

errorc = cl / (2.99*(10**8))

print errorc,cl




















