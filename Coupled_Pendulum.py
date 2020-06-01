# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:42:40 2019

@author: joshu
"""

from scipy.fftpack import fft, ifft
import numpy as np
from numpy import amax,amin,array
from math import log, pi
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.stats import linregress
from astropy import constants

data=ascii.read('spring3_h1_even_5deg')


time = []
pen1 = []
pen2 = []
for i in range(len(data['col1'])):
    if data['col1'][i] < 10:
        time.append(data['col1'][i])
        pen1.append(data['col2'][i])
        pen2.append(data['col3'][i])


plt.scatter(time,pen1)
plt.show()

from scipy.fftpack import fft
# Number of sample points
N = len(time)
# sample spacing
T = 0.0002
x = np.linspace(0.0, N*T, N)
y = array(pen1)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlim(0, 10)
plt.grid()
plt.show()

max_y = max(yf)  # Find the maximum y value  # Find the x value corresponding to the maximum y value
item_index = np.where(yf==max_y)
print xf[item_index]
