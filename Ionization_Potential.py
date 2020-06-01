# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:49:25 2019

@author: joshu
"""

import numpy as np
from numpy import linspace,mean,array,ndarray
from math import log, pi,sqrt
import matplotlib.pyplot as plt
import datetime
from astropy.io import ascii
from scipy.stats import linregress
from astropy import constants

dataB1T1=np.array(ascii.read('bulb1_t1.txt'))
dataB1T2=np.array(ascii.read('bulb1_t2.txt'))
dataB1T3=np.array(ascii.read('bulb1_t3.txt'))
dataB2T1=np.array(ascii.read('bulb2_t1.txt'))
dataB2T2=np.array(ascii.read('bulb2_t2.txt'))
dataB2T3=np.array(ascii.read('bulb2_t3.txt'))
dataB1fine=np.array(ascii.read('bulb1_fine.txt'))
dataB2fine=np.array(ascii.read('bulb2_fine.txt'))



iB1T1 = []
iB1T2 = []
iB1T3 = []
iB2T1 = []
iB2T2 = []
iB2T3 = []
iB1fine = []
iB2fine = []

###

for i in range(0,len(dataB1T1)):
    a = (dataB1T1['col2'][i])**(2.0/3.0)
    iB1T1.append(a)

for i in range(0,len(dataB1T2)):
    a = (dataB1T2['col2'][i])**(2.0/3.0)
    iB1T2.append(a)

for i in range(0,len(dataB1T3)):
    a = (dataB1T3['col2'][i])**(2.0/3.0)
    iB1T3.append(a)
    
for i in range(0,len(dataB2T1)):
    a = (dataB2T1['col2'][i])**(2.0/3.0)
    iB2T1.append(a)
    
for i in range(0,len(dataB2T2)):
    a = (dataB2T2['col2'][i])**(2.0/3.0)
    iB2T2.append(a)

for i in range(0,len(dataB2T3)):
    a = (dataB2T3['col2'][i])**(2.0/3.0)
    iB2T3.append(a)

for i in range(0,len(dataB1fine)):
    a = (dataB1fine['col2'][i])**(2.0/3.0)
    iB1fine.append(a)
    
for i in range(0,len(dataB2fine)):
    a = (dataB2fine['col2'][i])**(2.0/3.0)
    iB2fine.append(a)

###

polyxB1T1 = []
polyxB1T2 = []
polyxB1T3 = []
polyxB2T1 = []
polyxB2T2 = []
polyxB2T3 = []

polyyB1T1 = []
polyyB1T2 = []
polyyB1T3 = []
polyyB2T1 = []
polyyB2T2 = []
polyyB2T3 = []

###Trial 1

for i in range(0,len(dataB1T1['col1'])):
    a = dataB1T1['col1'][i]
    b = iB1T1[i]
    if a < 15.0:
        polyxB1T1.append(a)
        polyyB1T1.append(b)
    else:
        continue


b1t1polyfit = np.polyfit(polyxB1T1,polyyB1T1,1)
b1t1xpolyfit = linspace(-3,18,len(polyxB1T1))
b1t1ypolyfit = []


for k in range(0,len(polyxB1T1)):
    a = (b1t1polyfit[0])*(b1t1xpolyfit[k]) + b1t1polyfit[1]
    b1t1ypolyfit.append(a)
    
###Trial 2

for i in range(0,len(dataB1T2['col1'])):
    a = dataB1T2['col1'][i]
    b = iB1T2[i]
    if a < 15.0:
        polyxB1T2.append(a)
        polyyB1T2.append(b)
    else:
        continue


b1t2polyfit = np.polyfit(polyxB1T2,polyyB1T2,1)
b1t2xpolyfit = linspace(-3,18,len(polyxB1T2))
b1t2ypolyfit = []


for k in range(0,len(polyxB1T2)):
    a = (b1t2polyfit[0])*(b1t2xpolyfit[k]) + b1t2polyfit[1]
    b1t2ypolyfit.append(a)
    
###Trial 3

for i in range(0,len(dataB1T3['col1'])):
    a = dataB1T3['col1'][i]
    b = iB1T3[i]
    if a < 15.0:
        polyxB1T3.append(a)
        polyyB1T3.append(b)
    else:
        continue


b1t3polyfit = np.polyfit(polyxB1T3,polyyB1T3,1)
b1t3xpolyfit = linspace(-3,18,len(polyxB1T3))
b1t3ypolyfit = []


for k in range(0,len(polyxB1T3)):
    a = (b1t3polyfit[0])*(b1t3xpolyfit[k]) + b1t3polyfit[1]
    b1t3ypolyfit.append(a)

###Trial 1

for i in range(0,len(dataB2T1['col1'])):
    a = dataB2T1['col1'][i]
    b = iB2T1[i]
    if a < 15.0:
        polyxB2T1.append(a)
        polyyB2T1.append(b)
    else:
        continue


b2t1polyfit = np.polyfit(polyxB2T1,polyyB2T1,1)
b2t1xpolyfit = linspace(-3,18,len(polyxB2T1))
b2t1ypolyfit = []


for k in range(0,len(polyxB2T1)):
    a = (b2t1polyfit[0])*(b2t1xpolyfit[k]) + b2t1polyfit[1]
    b2t1ypolyfit.append(a)
    
###Trial 2

for i in range(0,len(dataB2T2['col1'])):
    a = dataB2T2['col1'][i]
    b = iB2T2[i]
    if a < 15.0:
        polyxB2T2.append(a)
        polyyB2T2.append(b)
    else:
        continue


b2t2polyfit = np.polyfit(polyxB2T2,polyyB2T2,1)
b2t2xpolyfit = linspace(-3,18,len(polyxB2T2))
b2t2ypolyfit = []


for k in range(0,len(polyxB2T2)):
    a = (b2t2polyfit[0])*(b2t2xpolyfit[k]) + b2t2polyfit[1]
    b2t2ypolyfit.append(a)
    
###Trial 3

for i in range(0,len(dataB2T3['col1'])):
    a = dataB2T3['col1'][i]
    b = iB2T3[i]
    if a < 15.0:
        polyxB2T3.append(a)
        polyyB2T3.append(b)
    else:
        continue


b2t3polyfit = np.polyfit(polyxB2T3,polyyB2T3,1)
b2t3xpolyfit = linspace(-3,18,len(polyxB2T3))
b2t3ypolyfit = []


for k in range(0,len(polyxB2T3)):
    a = (b2t3polyfit[0])*(b2t3xpolyfit[k]) + b2t3polyfit[1]
    b2t3ypolyfit.append(a)

###

f = plt.figure()


plt.title('I^(2/3) vs Voltage of Bulb 1')
plt.legend(loc='upper left')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.grid()
plt.scatter(dataB1T1['col1'],iB1T1,color = 'aqua',label='Trial 1')
#plt.plot(b1t1xpolyfit,b1t1ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb1_trial1.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()

plt.title('I^(2/3) vs Voltage of Bulb 1')
plt.legend(loc='upper left')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.grid()
plt.scatter(dataB1T2['col1'],iB1T2,color = 'salmon',label='Trial 2')
#plt.plot(b1t2xpolyfit,b1t2ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb1_trial2.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()

plt.title('I^(2/3) vs Voltage of Bulb 1')
plt.legend(loc='upper left')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.grid()
plt.scatter(dataB1T3['col1'],iB1T3,color = 'gold',label='Trial 2')
#plt.plot(b1t3xpolyfit,b1t3ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb1_trial3.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()

plt.scatter(dataB1fine['col1'],iB1fine,label='Zoomed')
plt.legend(loc='upper left')
plt.savefig("Bulb1.pdf", bbox_inches='tight')
plt.show()

plt.grid()
plt.scatter(dataB1fine['col1'],iB1fine,label='Zoomed')
plt.legend(loc='upper left')
plt.savefig("Bulb1close.pdf", bbox_inches='tight')
plt.show()




plt.grid()
plt.legend(loc='upper left')
plt.title('I^(2/3) vs Voltage of Bulb 2')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.scatter(dataB2T1['col1'],iB2T1,color = 'aqua',label='Trial 1')
#plt.plot(b2t1xpolyfit,b2t1ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb2_trial1.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()

plt.grid()
plt.title('I^(2/3) vs Voltage of Bulb 2')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.scatter(dataB2T2['col1'],iB2T2,color = 'salmon',label='Trial 2')
#plt.plot(b2t2xpolyfit,b2t2ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb2_trial2.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()

plt.grid()
plt.legend(loc='upper left')
plt.title('I^(2/3) vs Voltage of Bulb 2')
plt.ylabel('I^2/3')
plt.xlabel('Voltage')
plt.scatter(dataB2T3['col1'],iB2T3,color = 'gold',label='Trial 3')
#plt.plot(b2t3xpolyfit,b2t3ypolyfit,color = 'pink',label='Poly fit')
#plt.savefig("Bulb2_trial3.pdf", bbox_inches='tight')
#plt.legend(loc='upper left')
#plt.show()


plt.scatter(dataB2fine['col1'],iB2fine,label='Zoomed')
plt.legend(loc='upper left')
plt.savefig("Bulb2.pdf", bbox_inches='tight')
plt.show()

plt.grid()
plt.scatter(dataB2fine['col1'],iB2fine,label='Zoomed')
plt.legend(loc='upper left')
plt.savefig("Bulb2close.pdf", bbox_inches='tight')
plt.show()

b1t1errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1]
b1t2errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1]
b1t3errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1]
b1t1erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000003,0.000005,0.000015]
b1t2erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000005,0.000015]
b1t3erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000005,0.000015]


b2t1errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1]
b2t2errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1]
b2t3errv= [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1]
b2t1erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000003,0.000005,0.000015,0.000015]
b2t2erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000003,0.000005,0.000015]
b2t3erri = [0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.0000005,0.000003,0.000005,0.000015]


def xinter(m,b):
    return -1*(b) / m


xintb1t1 = xinter(b1t1polyfit[0],b1t1polyfit[1])
xintb1t2 = xinter(b1t2polyfit[0],b1t2polyfit[1])
xintb1t3 = xinter(b1t3polyfit[0],b1t3polyfit[1])

xintb2t1 = xinter(b2t1polyfit[0],b2t1polyfit[1])
xintb2t2 = xinter(b2t2polyfit[0],b2t2polyfit[1])
xintb2t3 = xinter(b2t3polyfit[0],b2t3polyfit[1])


b1 = array([xintb1t2,xintb1t1,xintb1t3])
b2 = array([xintb2t2,xintb2t1,xintb2t3])
avgxintb1 = mean(b1) 
avgxintb2 = mean(b2)  


ib1 = dataB1T1['col2'] + dataB1T2['col2'] + dataB1T2['col2']
ib2 = iB2T1 + iB2T2 + iB2T3

vb1 = ndarray.tolist(dataB1T1['col1']) + ndarray.tolist(dataB1T2['col1']) + ndarray.tolist(dataB1T3['col1'])
vb2 = ndarray.tolist(dataB2T1['col1']) + ndarray.tolist(dataB2T2['col1']) + ndarray.tolist(dataB2T3['col1'])


ib1err = b1t1erri + b1t2erri + b1t3erri
ib2err = b2t1erri + b2t2erri + b2t3erri

vb1err = b1t1errv + b1t2errv + b1t3errv
vb2err = b2t1errv + b2t2errv + b2t3errv


ib1lower = []
ib1upper = []
vb1lower = []
vb1upper = []

ib1t1lower = []
ib1t1upper = []
vb1t1lower = []
vb1t1upper = []

ib2lower = []
ib2upper = []
vb2lower = []
vb2upper = []

ib2t1lower = []
ib2t1upper = []
vb2t1lower = []
vb2t1upper = []


for i in range(0,len(b1t1erri)):
    ib1t1u = (dataB1T1['col2'][i] + b1t1erri[i])**(2.0/3.0)
    ib1t1l = (dataB1T1['col2'][i] - b1t1erri[i])**(2.0/3.0)
    vb1t1u = dataB1T1['col1'][i] + b1t1errv[i]
    vb1t1l = dataB1T1['col1'][i] - b1t1errv[i]
    if vb1t1u < 15.3:
        ib1t1upper.append(ib1t1u)
        ib1t1lower.append(ib1t1l)
        vb1t1upper.append(vb1t1u)
        vb1t1lower.append(vb1t1l)
    else:
        continue
    
for i in range(0,len(b2t1erri)):
    ib2t1u = (dataB2T1['col2'][i] + b2t1erri[i])**(2.0/3.0)
    ib2t1l = (dataB2T1['col2'][i] - b2t1erri[i])**(2.0/3.0)
    vb2t1u = dataB2T1['col1'][i] + b2t1errv[i]
    vb2t1l = dataB2T1['col1'][i] - b2t1errv[i]
    if vb2t1u < 15.3:
        ib2t1upper.append(ib2t1u)
        ib2t1lower.append(ib2t1l)
        vb2t1upper.append(vb2t1u)
        vb2t1lower.append(vb2t1l)
    else:
        continue


for i in range(0,len(ib1)):
    ib1u = (ib1[i] + ib1err[i])**(2.0/3.0)
    ib1l = (ib1[i] - ib1err[i])**(2.0/3.0)
    vb1u = vb1[i] + vb1err[i]
    vb1l = vb1[i] - vb1err[i]
    if vb1u < 15.0:
        ib1upper.append(ib1u)
        ib1lower.append(ib1l)
        vb1upper.append(vb1u)
        vb1lower.append(vb1l)
    else:
        continue
    



for i in range(0,len(ib2)):
    ib2u = (ib2[i] + ib2err[i])**(2.0/3.0)
    ib2l = (ib2[i] - ib2err[i])**(2.0/3.0)
    vb2u = vb2[i] + vb2err[i]
    vb2l = vb2[i] - vb2err[i]
    if vb2u < 15.0:
        ib2upper.append(ib2u)
        ib2lower.append(ib2l)
        vb2upper.append(vb2u)
        vb2lower.append(vb2l)
    else:
        continue


polyb1up = np.polyfit(vb1upper,ib1lower,1)
polyb1lo = np.polyfit(vb1lower,ib1upper,1)
polyb1t1up = np.polyfit(vb1t1upper,ib1t1lower,1)
polyb1t1lo = np.polyfit(vb1t1lower,ib1t1upper,1)
polyb2up = np.polyfit(vb2upper,ib2upper,1)
polyb2lo = np.polyfit(vb2lower,ib2lower,1)
polyb2t1up = np.polyfit(vb2t1upper,ib2t1lower,1)
polyb2t1lo = np.polyfit(vb2t1lower,ib2t1upper,1)

#print polyb1up[0], polyb1lo[0]
#print polyb2up[0], polyb2lo[0]
#print polyb1t1up[0], polyb1t1lo[0]
#print polyb2t1up[0], polyb2t1lo[0]

b1slopeyerr = ((polyb1up[0] - polyb1lo[0]) / 2.0 )
b1t1slopeyerr = ((polyb1t1up[0] - polyb1t1lo[0]) / 2.0 )
b2slopeyerr = ((polyb2up[0] - polyb2lo[0]) / 2.0 )
b2t1slopeyerr = ((polyb2t1up[0] - polyb2t1lo[0]) / 2.0 )
b1t1intererr = ((polyb1t1up[1] - polyb1t1lo[1]) / 2.0 )
b2t1intererr = ((polyb2t1up[1] - polyb2t1lo[1]) / 2.0 )

#print b1t1intererr,b2t1intererr

#print b1t1slopeyerr
#print b2t1slopeyerr

b1yintercepterror = [0.0002,0.0002,0.0002]
b2yintercepterror = [0.0002,0.0002,0.0002]

b1slopeerror = [10**-7,10**-7,10**-7]
b2slopeerror = [10**-7,10**-7,10**-7]

b1slope = [b1t1polyfit[0],b1t2polyfit[0],b1t3polyfit[0]]
b1yintercept = [b1t1polyfit[1],b1t2polyfit[1],b1t3polyfit[1]]
b2slope = [b2t1polyfit[0],b2t2polyfit[0],b2t3polyfit[0]]
b2yintercept = [b2t1polyfit[1],b2t2polyfit[1],b2t3polyfit[1]]


def multierror(value,y,ye,s,se):
    return  value* (sqrt(((ye/y)*100)**2 + ((se/s)*100)**2) / 100.0 )


b1t1xinterror = multierror(xintb1t1,b1yintercept[0],b1yintercepterror[0],b1slope[0],b1slopeerror[0])
b1t2xinterror = multierror(xintb1t2,b1yintercept[1],b1yintercepterror[1],b1slope[1],b1slopeerror[1])
#print b1t1xinterror
b2t1xinterror = multierror(xintb2t1,b2yintercept[0],b2yintercepterror[0],b2slope[0],b2slopeerror[0])
#print b2t1xinterror



b1xinterror = sqrt(3*((0.05)**2))
b2xinterror = sqrt(3*((0.03)**2))

print avgxintb1,avgxintb2
print b1xinterror,b2xinterror



b1break = [14.6,14.9,14.1]
b1breakerr = sqrt((3*(0.05)**2))
b2break = [15,14.7,14.8]

print mean(b1break),mean(b2break),b1breakerr

print sqrt(0.08**2 + 0.09**2)
print sqrt(0.08**2 + 0.05**2)








