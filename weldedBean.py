#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:06:49 2018

@author: Kelvin Milach & Gabriel Mouzella
"""

import numpy as np
from operator import itemgetter

P = 6000
L = 14
E = 30000000
G = 12000000


#******************************************************************************#
#FUNÇÕES AUXILIARES                                                            #
#******************************************************************************#

def Pc(x):
  x1, x2, x3, x4 = x
  return ((4.013*E*np.sqrt((x3**2)*(x4**6)))/(L^2))*(1 - ((x3/(2*L))*(np.sqrt(E/(4*G)))))

def delta(x):
  x1, x2, x3, x4 = x
  return (4*P*(L**3))/(E*x4*(x3**3))

def sigma(x):
  x1, x2, x3, x4 = x
  return (6*P*L)/(x4*(x3**2))

def J(x):
  x1, x2, x3, x4 = x
  return 2*(np.sqrt(2)*x1*x2*np.sum([(x2**2)/12, ((x1 + x3)/2)**2]))

def R(x):
  x1, x2, x3, x4 = x
  return np.sqrt((x2**2)/4 + x1 + x3)

def Q(x):
  x1, x2, x3, x4 = x
  return P*(L + x2/2)

def tau1(x):
  x1, x2, x3, x4 = x
  return P/(x1*x2*np.sqrt(2))

def tau2(x):
  return (Q(x)*R(x))/J(x)

def tau(x):
  x1, x2, x3, x4 = x
  return np.sqrt(tau1(x)**2 + 2*tau1(x)*tau2(x)*(x2/(2*R(x))) + tau2(x)**2)

#******************************************************************************#
# RESTRIÇÕES                                                                   #
#******************************************************************************#
  
def g1(x):
  return (tau(x) - 13600)

def g2(x):
  return (sigma(x) - 30000)

def g3(x):
  x1, x2, x3, x4 = x
  return (x1 - x4)

def g4(x):
  x1, x2, x3, x4 = x
  return (0.10471*(x1**2) + (0.04811*x3*x4*(14.0 - x2)) - 5)

def g5(x):
  x1, x2, x3, x4 = x
  return (0.125 - x1)

def g6(x):
  return (delta(x) - 0.25)

def g7(x):
  return (P - Pc(x))

def constr_wbd(x):        
  return [0 - g1(x), 0 - g2(x), 0 - g3(x), 0 - g4(x), 0 - g5(x), 0 - g6(x), 0 - g7(x)]

#******************************************************************************#
# Constições
#******************************************************************************#
#--------------------------------------------------------------

##FUNÇÃO ^C
def conta_restr(x): #esta função conta o número de restrições que foram infringidas
  num_restr = 0
  for i in constr_wbd(x):
    if i < (-1e-3):
      num_restr += 1
  return num_restr

##
def soma_restr(x):  #esta função faz a soma de g1(x), g2(x), ... g7(x) para dado x = (x1, x2, x3, x4)
  return sum(constr_wbd(x))


##FUNÇÃO vj(vetor x)
def distance_v(x):  #esta funcao retorna uma lista de 7 argumentos (7 restrições), onde retorna 0 para quando a restrição n é violada
                    #ou o valor da função de restrição violada
  return [max(0, g1(x)), max(0, g2(x)), max(0, g3(x)), max(0, g4(x)), max(0, g5(x)), max(0, g6(x)), max(0, g7(x))]

##
def count_ICS(pop): #essa função retorna o número de soluções que infrigem alguma restrição
  ICS = 0
  for x in pop:
    if conta_restr(x) > 0:
      ICS += 1
  return ICS

#SUGESTÃO DO CASTOR:
# criar listas separadas de soluções possíveis e não possiveis
# em cada lista, modificar os elementos para tuplas com 3 novas caracteristicas:
# 1 - o valor da função fit (f1), 2 - nº de restrições infringidas, 3 - distancia absoluta do erro (sum(distance_v(x)))

def list_FCS(pop):  #cria uma lista com as soluções FCS (possíveis) de uma população
  y_fcs = []
  for x in pop:
    if conta_restr(x) == 0:
      y_fcs.append(x)
  
  return ranking_FCS(y_fcs)

def list_ICS(pop):  #cria uma lista com as soluções ICS (não possíveis) de uma população
  y_ics = []
  for x in pop:
    if conta_restr(x) > 0:
      y_ics.append(x)
  
  return ranking_ICS(y_ics)

def ranking_FCS(y_fcs):   #faz o ranking da parte da população que é feasible (FCS)
  for fcs in y_fcs:
    fcs.append(fitFunction(fcs))
  return sorted(y_fcs,key=itemgetter(4))

def ranking_ICS(y_ics):   #faz o ranking da parte da população que é feasible (FCS)
  for ics in y_ics:
    n_restr_infrin = conta_restr(ics) #número de restrições infringidas
    soma_restr_infrin = abs(soma_restr(ics)) #pegamos o módulo da soma das violações de restrição
    
    ics.append(n_restr_infrin)
    ics.append(soma_restr_infrin)
    sorted(y_ics,key=itemgetter(4, 5))
    
  return sorted(y_ics,key=itemgetter(4, 5))
#******************************************************************************#
# FIT FUNCTION                                                                 #
#******************************************************************************#
  
def fitFunction(x):
    x1, x2, x3, x4 = x
    return (1.10471*x2*x1**2 + 0.04811*x3*x4*(14.0 + x2))

def boundaries():
    lb = [0.1, 0.1, 0.1, 0.1]
    ub = [2, 10, 10, 2]
    return lb,ub

def optimal():
    return [0.205830, 3.468338, 9.036624, 0.205730] #solução ÓTIMA encontrada no paper