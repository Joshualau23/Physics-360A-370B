# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 16:56:32 2018

@author: Josh
"""

from math import sqrt,log

background = [0.017,0.016,0.017,0.016,0.014,0.013,0.012,0.012,0.012,0.012]
beam = [.266,.265,.265,.264,.263,.270,.26,.26,.261,.26]
ampbeam = [.281,.291,.289,.276,.275,.274,.286,.273,.289,.273]
pm = [0.0014,0.0014,0.0014,0.0014,0.0014,0.0014,0.0014,0.0014,0.0014,0.0014]

pout = []
pin = []

for i in range(0,10):
    a = ampbeam[i] - background[i]
    b = beam[i] - background[i]
    pout.append(a)
    pin.append(b)

gerror = []
g = []

for k in range(0,10):
    c = sqrt(pout[k] / pin[k])
    G = pout[k] / pin[k]
    outper = 100*(pm[k] / pout[k] )
    inper = 100*(pm[k] / pin[k] )
    pererror = (1.0/2.0)*sqrt( outper**2 + inper**2)
    error = G * (pererror / 100)
    g.append(c)
    gerror.append(error)
    
alpha = []

avg = sum(g) / float(len(g))

for i in range(0,10):
    al = log(g[i]) / 0.395
    alpha.append(al)
    
background2 = [0.005,0.006,0.006,0.008,0.008,0.006,0.005,0.004,0.005,0.004]
beam2 = [.68,.682,.682,.682,.682,.683,.681,.683,.682,.682]

laser = []

for i in range(0,10):
    a = beam2[i] - background2[i]
    laser.append(a)


avg2 = sum(laser) / float(len(laser))
print avg2
