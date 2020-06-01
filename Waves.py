# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:09:04 2019

@author: joshu
"""

from scipy.fftpack import fft, ifft
import numpy as np
from numpy import amax,amin,array
from math import log, pi,tan,sqrt
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.stats import linregress
from astropy import constants

data=ascii.read('closed.txt')
data2=ascii.read('open.txt')


hertz = array(data['col1'])
hertzerr = array(data['col7'])

closedV = data['col2']
closedVerr = data['col3']
closedI = data['col4']
closedIerr = data['col5']

openV = data2['col2']
openVerr = data2['col3']
openI = data2['col4']
openIerr = data2['col5']

z = closedV / closedI
z2 = openV / openI

y = []
y2 = []

for i in range(0,len(hertz)):
    c = tan(z[i])
    o = tan(z2[i])
    y.append(c)
    y2.append(o)
    
zerr = []
z2err = [] 
for i in range(0,len(hertz)):
    a = z[i] * sqrt(((closedVerr[i] / closedV[i])*100.0 )**2.0 + ((closedIerr[i] / closedI[i])*100.0 )**2.0) / 100
    b = z2[i] * sqrt(((openVerr[i] / openV[i])*100.0 )**2.0 + ((openIerr[i] / openI[i])*100.0 )**2.0) / 100
    zerr.append(a)
    z2err.append(b)

fig,ax=plt.subplots()

plt.plot(hertz,z,color = 'Gold',label='Closed')
plt.errorbar(hertz,z,xerr=hertzerr,yerr=zerr, linestyle="None",color='sienna',label='Closed Error')
plt.plot(hertz,z2,color = 'deepskyblue',label='Open')
plt.errorbar(hertz,z2,xerr=hertzerr,yerr=z2err, linestyle="None",color='midnightblue',label='Open Error')
plt.legend(loc='upper left')
plt.title('Input Impedence versus frequency')
plt.ylabel('Zin (V/mA)')
plt.xlabel('Frequency (Mhz)')
plt.xticks(np.arange(min(hertz), max(hertz)+1, 1.0))
plt.grid()
plt.savefig('partb')
plt.show()

###

closedmin = [1.6,3.3,4.9,6.5]
openmin = [2.4,4.1,5.7,7.3]

closedreson = []
openreson = []

for i in range(0,len(closedmin)-1):
    a = closedmin[i+1] - closedmin[i]
    b = openmin[i+1] - openmin[i]
    closedreson.append(a)
    openreson.append(b)

print closedreson,openreson


data3=ascii.read('scope_1.txt')
time = data3['col1'] 
voltage = data3['col2']
current = data3['col3']
print time

data4=ascii.read('scope_3.txt')
time2 = data4['col1']
voltage2 = data4['col2']
current2 = data4['col3']

plt.plot(time* 10e6,voltage,color = 'Gold',label='Voltage')
plt.xticks(np.arange(min(time), max(time)))
plt.legend(loc='upper left')
plt.title('Voltage versus time')
plt.ylabel('Voltage (V)')
plt.xlabel('time (10^-6s)')
plt.grid()
plt.savefig('partc')
plt.show()

plt.plot(time2,voltage2,color = 'Gold',label='Closed')
plt.plot(time2,current2,color = 'blue',label='Closed')
plt.show()

minvol = min(voltage)
maxvol = max(voltage)
deltat = []
for i in range(0,len(time)):
    if voltage[i] == maxvol or voltage[i] == minvol:
        deltat.append(time[i])
    else:
        continue
print deltat

t = deltat[1] - deltat[0]
print t
speed = 120 / t
print speed
vfactor =  speed / 299792458
print vfactor

dconstant = (1 / vfactor**2)
print dconstant
        


mincur = min(current2)
maxcur = max(current2)
deltat2 = []
for i in range(0,len(time2)):
    if current2[i] == maxcur or current2[i] == mincur:
        deltat2.append(time2[i])
    else:
        continue
print deltat2

t2 = deltat2[1] - deltat2[0]
speed2 = 120 / t2
print speed2
vfactor2 =  speed2 / 299792458
print vfactor2

dconstant2 = (1 / vfactor2**2)
print dconstant2
    

LC = 2.17e-5 * 1.8e-9
vp = 1 / sqrt(LC)
resonfreq = 1 / (2*pi*sqrt(LC))
print resonfreq / 1000000.0

timerr = sqrt(2*(0.02e-7)**2)
print timerr
verr = sqrt((timerr*100/t)**2 + (0.1*100/120.0)**2 ) / 100
print verr * 0.5 * dconstant

Lerr = 2.64e-7
Cerr = 2.24e-11
c = 299792458
LCerr = sqrt((Lerr *100 / 2.17e-5)**2 + (Cerr*100 / 1.8e-9)**2)  / 100 * 0.5 
epsilon = LC*(299792458)**2
print (0.75*c) /(2*60) / 1000000






