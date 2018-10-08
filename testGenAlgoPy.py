#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:36:07 2018

@author: deboismo
"""

import gap as ga
import weldedBean as wb

## PARAMETROS PSO
swarmsize = 100
omega = 0.3
phip = 0.5
phig = 0.5
minstep = 1e-8
minfunc = 1e-8
maxiter = 1000
debug= True

## PARAMETROS GA
epochs = 700
crossover = 0.25
mutation = 0.01

fitFunction = wb.fitFunction
lb,ub = wb.boundaries()
constr = wb.constr_wbd
optimal = wb.optimal()
vecBestPossibles = wb.list_FCS
vecWorstPossibles = wb.list_ICS

best = ga.GA(fitFunction, lb, ub, populationSize=1000, epochs=500, 
       crossover=0.7, mutation=0.05, verbose = 0, numberTrainings = 10)
