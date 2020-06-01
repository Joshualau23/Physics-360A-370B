# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:15:32 2018

@author: Josh
"""

from astropy.io import ascii
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math


### Part A

voltage712 = []
count712 = []
voltage712p = []
count712p = []


voltage712 = ascii.read('712.txt')['Voltage']
count712 = ascii.read('712.txt')['Count']


for i in range(2,13):
    x = voltage712[i]
    y = count712[i]
    voltage712p.append(x)
    count712p.append(y)

x = np.arange(350,618)
b, m = polyfit(voltage712p, count712p, 1)


plt.plot(voltage712,count712)
plt.scatter(voltage712,count712,marker=".",color = 'k')
plt.plot(x, b + m * x, '--',color = 'r')
plt.title('LND 712 Geiger Tube')
plt.xlabel('Voltage (v)')
plt.ylabel('Count')
plt.xticks(np.arange(250,700 , step=40))
plt.xlim(xmin=250,xmax=700)
plt.yticks(np.arange(350, 500, step=40))
plt.ylim(ymin=350,ymax=500)
plt.savefig('parta712.pdf')



### LND 314

voltage314 = []
count314 = []
voltage314p = []
count314p = []


voltage314 = ascii.read('314.txt')['Voltage']
count314 = ascii.read('314.txt')['Count']


for i in range(7,19):
    x = voltage314[i]
    y = count314[i]
    voltage314p.append(x)
    count314p.append(y)


x2 = np.arange(400,625)
b2, m2 = polyfit(voltage314p, count314p, 1)


plt.plot(voltage314,count314,color = 'b')
plt.scatter(voltage314,count314,marker=".",color = 'k')
plt.plot(x2, b2 + m2 * x2, '--',color = 'r')
plt.title('LND 72314 Geiger Tube')
plt.xlabel('Voltage (v)')
plt.ylabel('Count')
plt.xticks(np.arange(250,700 , step=40))
plt.xlim(xmin=250,xmax=700)
plt.yticks(np.arange(600, 1200, step=50))
plt.ylim(ymin=600,ymax=1200)
plt.savefig('parta314.pdf')


print m,m2

#PartB

from astropy.io import ascii
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

count10 = ascii.read('314_500_10pop.txt')['count']
count100 = ascii.read('314_500_100pop.txt')['count']
count300 = ascii.read('314_500_300pop.txt')['count']
count500 = ascii.read('314_500_500pop.txt')['count']

x500 = 0
y500 = 0
for i in range(0,len(count500)):
    if count500[i] > 23.54 or count500[i] < 15.18:
        x500 += 1
    if count500[i] > 27.72 or count500[i] < 11:
        y500 += 1
print x500,y500

x300 = 0
y300 = 0
for i in range(0,len(count300)):
    if count300[i] > 23.77 or count300[i] < 15.55:
        x300 += 1
    if count300[i] > 27.88 or count300[i] < 11.44:
        y300 += 1
print x300,y300

x100 = 0
y100 = 0
for i in range(0,len(count100)):
    if count100[i] > 23 or count100[i] < 15.26:
        x100 += 1
    if count100[i] > 26.87 or count100[i] < 11.39:
        y100 += 1
print x100,y100

x10 = 0
y10 = 0
for i in range(0,len(count10)):
    if count10[i] > 24.99 or count10[i] < 16.81:
        x10 += 1
    if count10[i] > 29.08 or count10[i] < 12.72:
        y10 += 1
print x10,y10

    

plt.hist(count10, bins=14)
plt.title('Sample Size of 10')
plt.xlabel('Count')
plt.ylabel('# of Occurances')
plt.ylim(ymin=0,ymax=3)
plt.savefig('partc10.pdf')
plt.show()
plt.hist(count100, bins=25)
plt.title('Sample Size of 100')
plt.xlabel('Count')
plt.ylabel('# of Occurances')
plt.ylim(ymin=0,ymax=15)
plt.savefig('partc100.pdf')
plt.show()
plt.hist(count300, bins=25)
plt.title('Sample Size of 300')
plt.xlabel('Count')
plt.ylabel('# of Occurances')
plt.ylim(ymin=0,ymax=70)
plt.savefig('partc300.pdf')
plt.show()
plt.hist(count500, bins=25)
plt.title('Sample Size of 500')
plt.xlabel('Count')
plt.ylabel('# of Occurances')
plt.ylim(ymin=0,ymax=55)
plt.savefig('partc500.pdf')
plt.show()
