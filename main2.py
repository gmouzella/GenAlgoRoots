#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:45:54 2018

@author: deboismo
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import pyswarm as ps
import matplotlib.pyplot as plt
import gap
import math

#**************************************************************************
def neuralNetwork1(w):
    x = np.linspace(-4,4,100, endpoint = True)
    p1 = p1Function(x)
    mse = mean_squared_error(p1,polinomialFunction1(w,x))
    return math.sqrt(mse) 

#**************************************************************************
def neuralNetwork2(w):
    x = np.linspace(-4,4,100, endpoint = True)
    p2 = p2Function(x)
    mse = mean_squared_error(p2,polinomialFunction2(w,x))
    return math.sqrt(mse) 

#**************************************************************************
def polinomialFunction1(w,x):
    p1hat = np.ones(len(x))
    x = np.reshape(x,(len(x),1))
    temp = x-w
    
    for i in range(len(w)):    p1hat = p1hat*temp[:,i]

    return p1hat
    
    
#**************************************************************************
def polinomialFunction2(w,x):
    p2hat = np.ones(len(x))
    x = np.reshape(x,(len(x),1))
    temp = x-w
    
    for i in range(len(w)):    p2hat = p2hat*temp[:,i]
  
    return p2hat

#**************************************************************************
def p1Function(x):
    return np.array(x**7 - 5*x**6 - 37*x**5 + 120*x**4 + 223*x**3 - 639*x**2 + 240*x + 88)

#**************************************************************************
def p2Function(x):
  return np.array(2*x**6 - 4*x**5 - 50*x**4 + 70*x**3 + 246*x**2 - 30*x - 106)

#**************************************************************************
def constraint1(x):
    return [0 - x[0], 0 - x[1], 0 - x[2], 0 - x[3], 0 - x[4], 0 - x[5], 0 - x[6]]
#**************************************************************************
def constraint2(x):
    return [0 - x[0], 0 - x[1], 0 - x[2], 0 - x[3], 0 - x[4], 0 - x[5]]
#**************************************************************************

swarmsize = 300
#omega = 0.6
#phip = 0.75
#phig = 0.57
#minstep = 1e-20
#minfunc = 1e-20
maxiter = 200
#debug = True

raizes = [-4.23157, -1.69869, -0.697938, 0.685078, 3.37775, 4.56537]

global lb; global ub;

func = int(input('Qual função deseja executar:'\
             '\n\t 1 - f(x) = x**7 - 5*x**6 - 37*x**5 + 120*x**4 + 223*x**3 - 639*x**2 + 240*x + 88'\
             '\n\t 2 - f(x) = 2*x**6 - 4*x**5 - 50*x**4 + 70*x**3 + 246*x**2 - 30*x - 106' \
             '\nResposta: '))


if func == 1:
    fitFunction = neuralNetwork1
    lb, ub = np.ones(7)*(-8), np.ones(7)*(8)    
    preal = p1Function
    phat = polinomialFunction1
    const = constraint1

elif func == 2:
    fitFunction = neuralNetwork2
    lb, ub = np.ones(6)*(-5), np.ones(6)*(5)    
    preal = p2Function
    phat = polinomialFunction2
    const = constraint2


#xopt, fopt = ps.pso(fitFunction, lb, ub, swarmsize = swarmsize, omega = omega,
#                            phip = phip, phig = phig, minstep = minstep, minfunc = minfunc,
#                            maxiter = maxiter, debug = debug)

best_individuals = gap.GA(fitFunction,lb,ub,populationSize = swarmsize, generation = maxiter, 
       crossover = 0.5, mutation = 0.05, numberTrainings = 1, constraints = const)

x = np.linspace(-5,5,1000, endpoint = True)
pReal = preal(x)
pHat = phat(np.array(best_individuals[0][0]),x)

fig = plt.figure(figsize = (15,10))
plt.plot(x,pReal)
plt.plot(x,pHat)
fig.show()












