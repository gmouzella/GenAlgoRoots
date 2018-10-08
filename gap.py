'''
Copyright (c) 2018 Gabriel Mouzella Silva & Kelvin Ferreira Milach

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

__author__ = "Gabriel Mouzella Silva & Kelvin Ferreira Milach"
__copyright__ = "Copyright (C) 2018 Gabriel Mouzella Silva & Kelvin Ferreira Milach"
__license__ = "Public Domain"
__version__ = "1.0"
__email__ = "<gmouzella@gmail.com>, <milach.kelvin@gmail.com>"


import math
import numpy as np
import random
import sys
from operator import itemgetter, add
#from progress.bar import ProgressBar



'''
GA characteristics:
    binary tournament (select the best between 2 indivuduals)
    arithmetic crossover
    Gaussian mutation (normal distribution)

Brief introduction on the functions in this module:
    
    "public" function:
        
        GA                - the main function from the module, that create and
                            execute the genetic algorithm code.
    
    "private" functions:
    
        _createPopulation - initialize the population of individuals using a 
                            uniform distribution.
        
        _selectIndividual - select the winner of the binary tournament
        
        _binaryTournament - randomly select 2 individuals from the population
                           and compare them.
                           
        _arithXover       - make the arithmetic crossover between parents.
        
        _mutationFunction - mutate the individual in the case of it happening.
                           Possible distribuitons: normal, poisson and uniform.
                           
        _nextGeneration   - breads the parents to generate a new generation of 
                           individuals.
                           
        _countConstraint  - count the number of contraints infringed.
        
        _sumConstraint    - sum the values of all constraints infringed.
        
        _listIndividuals  - rank the individuals into the possible and not_possible
                            list.
        
        _get_best         - return the best of all individuals
        
        _get_worst        - return the worst of all individuals
'''



################################################################################
# CREATE INITAIL POPULATION                                            PRIVATE #
################################################################################
def _createPopulation(fitFunction, lowerBoundary, upperBoundary, constraint, 
                      populationSize = 100):
  '''
  initialize the population. Called at the beggining of GA function.
  
  inputs:
      
      fitFunction: the fit function that needs to be optimized, with a 1d-array
                   as input and returns a single value.
                
      lowerBoundary: 1d-array with the lenght of the number of varibles trying to
                     optimize, with the lowest values possible to the search for
                     each variable.
    
      upperBoundary: 1d-array with the lenght of the number of varibles trying to
                     optimize, with the highest values possible to the search for
                     each variable.
                     
      constraint: constraint function 

  output:
      
      population: list with all the individuals created.
      
  '''
  
  #initialize list of individuals
  population = []
  
  #initialize list of possible solutions for each individual
  listSolution = [] 
  
  for _ in range(populationSize):
    
    # create a list with values for the variables to be optimizes
    solution = []
    
    # create randomically the values for the variables of each individual
    # using uniform distribution and within the boundaries and constraints
    [solution.append(random.uniform(lowerBoundary[j],upperBoundary[j])) 
    for j in range(len(lowerBoundary))]
    
    # append all values
    listSolution.append(solution)
  
  for k in range(populationSize):
    
    # create the individuals
    individuo = [listSolution[k], fitFunction(listSolution[k]), _countConstraint(listSolution[k],
                 constraint), _sumConstraint(listSolution[k], constraint)]
    
    # initialize all individuals in the initial position    
    population.append(individuo)
    
  return population # return the list of individuals


################################################################################
# SELECT THE INDIVIDUAL FOR THE BINARY TOURNAMENT                      PRIVATE #
################################################################################
def _selectIndividual(individual1, individual2):
  '''
  select the best individual compared in pairs for the binary tournament. the
  rules that decide the best indivual are:
      
      1 - if at least one individuals is viable, and does not infringe any constraint:
          
          1.1 - if both are viable wins the one with better fit function value
          
          1.2 - if only one is viable, he is the winner.
          
      2 - if both individuals are not viable, wins who has the least number of
          infractions
    
      3 - if the number of infractions are equal, the one with lowest sum of
          infractions wins
  
  inputs:
      
      individual1, individual2: vector containing all the values for the individual
      
  output:
      
      returns the best between the 2 individuals
  '''
  
  # Step 1.1:
  if (individual1[2]==0 and individual2[2]==0):
    
    if individual1[1]<individual2[1]:   return individual1
    
    else:   return individual2
  
  # Step 1.2
  elif ((individual1[2]==0 and individual2[2]>0) or (individual1[2]>0 and individual2[2]==0)):
    
    if (individual1[2]==0):    return individual1
    
    else:   return individual2
  
  else:
    
    # Step 2
    if (individual1[2] != individual2[2]):
      
      if individual1[2]<individual2[2]:     return individual1
      
      else:    return individual2
      
    # Step 3
    else:
      
      if individual1[3]<individual2[3]:   return individual1
      
      else:   return individual2


################################################################################
# SELECT THE INDIVIDUAL FOR THE BINARY TOURNAMENT                      PRIVATE #
################################################################################
def _binaryTournament(population):
  '''
  randomly select 2 individuals and compare between them.
  
  inputs:
      
      population: list with all the values for each individual
      
  output:
      
      return list with the selected parents to breed
  '''
  
  parents = []
  
  for _ in range(len(population)):
    
    firstParent, secondParent = random.sample(population, 2)
    
    parents.append(_selectIndividual(firstParent, secondParent))
  
  return parents


################################################################################
# ARITHMETIC CROSSOVER                                                 PRIVATE #
################################################################################
def _arithXover(firstParent, secondParent, fit, constraint, beta=0.3):
  '''
  Make the arithmetic crossover of the parents in order to generate the sons.
  
  For a given value of beta, the crossover result is:
      
      cromossome1 = beta*firstParent + (1-beta)*secondParent
      
      cromossome1 = (1-beta)*firstParent + beta*secondParent

  inputs:
      
      firstParent, secondParent: list with the data for the parent to breed
      
      fit: fitFunction to be optimized
      
      constraint: constraint functions
      
      beta: value between 0 and 1
      
  output:
      
      return both sons resulted from the breed of the parents
  '''
  
  cromossome1 = list(map(add, [beta*elm for elm in firstParent[0]], [(1-beta)*elm for elm in secondParent[0]]))

  firstSon = [cromossome1, fit(cromossome1), _countConstraint(cromossome1, constraint), _sumConstraint(cromossome1, constraint)]
  
  
  cromossome2 = list(map(add, [beta*elm for elm in secondParent[0]], [(1-beta)*elm for elm in firstParent[0]]))
  
  secondSon = [cromossome2, fit(cromossome2), _countConstraint(cromossome2, constraint), _sumConstraint(cromossome2, constraint)]
  
  
  return firstSon, secondSon


################################################################################
# ARITHMETIC CROSSOVER                                                 PRIVATE #
################################################################################
def _mutationFunction(individual, lowerBoundarie, upperBoundarie, mutationDistribution = 'normal', mutationProbability=0.01, verbose = 0):
  '''
  This function realize the mutation of individuals given a probability.
  
  The gene to be changed is selected randomly.
  
  The default and recomended distribution to the mutation operation is the normal or gaussian
  distribution. However it is given here the possibility of using also the uniform and poisson
  distribution to perform the mutation.
  
  inputs:
      
      individual: singular individual of the entire population
      
      lowerBoundarie: 1d-array with the minimum boundaries of the variables
      
      upperBoundarie: 1d-array with the maximum boundaries of the variables
      
      mutationDistribution: distribution to be used to perform the mutation
      
      mutationProbability: probability of mutation occurency
      
      verbose: show the progress of the training. Default 0 (show nothing)
      
  output:
      
      returns the individual, with or without the mutation
  '''
  
  
  if random.random() <= mutationProbability:
    #if the mutation happens then the gene will be selected randomly
    
    #selected gene
    gene = random.randint(0, len(individual[0])-1)
    
    # normal distribution
    if (mutationDistribution == 'normal' or mutationDistribution == 'gaussian'):  
        
        #standard deviation for the mutation 
        sigma = (upperBoundarie[gene] - lowerBoundarie[gene])/10
        
        individual[0][gene] = min(max(np.random.normal(0, sigma), upperBoundarie[gene]), lowerBoundarie[gene])
    
    # uniform distribution
    elif (mutationDistribution == 'uniform'):  
        
        individual[0][gene] = min(max(np.random.uniform(), upperBoundarie[gene]), lowerBoundarie[gene])
    
    # poisson distribution
    elif (mutationDistribution == 'poisson'):  
        
        individual[0][gene] = min(max(np# cria a proxima geração
    .random.poisson(), upperBoundarie[gene]), lowerBoundarie[gene])
    
    if(verbose == 1):    print("MUTATION HAPPENED!!")
    
    return individual
  
  # if mutation didn't happened
  else:
    return individual

################################################################################
# Next Generation                                                      PRIVATE #
################################################################################
def _nextGeneration(listParents, fit, constraint, lowerBoundarie, upperBoundarie, 
                    crossover=0.4, mutate=0.01, verbose = 0, populationSize = 50):
  '''
  Creates the next generation of individuals using a list of selected parents
  and the crossover (function: _arithXover) described above.
  
  inputs:
      
      listParents: list containing the parents selected
      
      fit: fit function to be optimized
      
      constraint: constraints to be applied
      
      lowerBoundarie: 1d-array with the minimum boundaries of the variables
      
      upperBoundarie: 1d-array with the maximum boundaries of the variables
      
      crossover: value between 0-1 to in wich the crossover will happen
      
      mutate: probability of mutation
      
      verbose: show the progress of the training. Default 0 (show nothing)
      
      popultaionSize: size of total population
      
  outputs:
      
      list containing the individuals generated for the next generation
      
  '''

  # list to be filled with the next generation  
  listSons = []
  
  repetitionFactor = int(populationSize/len(listParents))
  
  for _ in range(repetitionFactor):
      
      for i in range(0, len(listParents), 2):
        
        # arithmetic crossover of parents
        firstSon = _arithXover(listParents[i], listParents[i+1], fit, constraint, beta=crossover)[0]
        # mutation
        firstSon = _mutationFunction(firstSon, lowerBoundarie, upperBoundarie, 'normal', mutate, verbose)
        
        # arithmetic crossover of parents
        secondSon = _arithXover(listParents[i], listParents[i+1], fit, constraint, beta=crossover)[1]
        # mutation
        secondSon = _mutationFunction(secondSon, lowerBoundarie, upperBoundarie, 'normal', mutate, verbose)
        
        listSons.append(firstSon),listSons.append(secondSon)
  
      listParents = random.shuffle(listParents)
    
  return listSons

################################################################################
# Count Constraint                                                     PRIVATE #
################################################################################
def _countConstraint(variableVector, constriction): #esta função conta o número de restrições que foram infringidas
    '''
    Count the number of constraints infringed.
    
    inputs:
        
        variableVector: vector containing the values of the variables to be optimzed
        
        constriction: 1d-array containing the constraint functions
    
    output:
        
        return an integer containing the number of constrictions infringed
    '''
    numConstraints = 0
  
    if (constriction == []): return 0
  
    else: return [numConstraints+1 if value < 0 else 0 for value in constriction(variableVector)][-1]


################################################################################
# Sum Constraints                                                      PRIVATE #
################################################################################
def _sumConstraint(variableVector, constriction):  
    '''
    Sums the values of the infringed constraints
    
    inputs:
        
        variableVector: vector containing the values of the variables to be optimzed
        
        constriction: 1d-array containing the constraint functions
    
    output:
        
        return the sum of all the infringed constriction values
    '''
    if (constriction == []): return 0
    
    else: return math.fsum([abs(elm) for elm in constriction(variableVector)])

################################################################################
# List individuals                                                     PRIVATE #
################################################################################
def _listIndividuals(population, ranked = True):  #cria uma lista com as soluções FCS (possíveis) de uma população  
  '''
  Creates lists of possibles and not possibles (due to constraint infrictions)
  individuals
  
  inputs:
      
      population: list with the entire population
      
      ranked: (boolean) decide if the list should be ranked or not. Default = True
      
  output:
      
      possibleIndividuals: list with all possible individuals
      
      notPossibleIndividuals: list with all impossible individuals
  '''

  # select all the possible individuals into a list
  possibleIndividuals = [individual for individual in population if individual[2] == 0]
  
  # select all the possible individuals into a list
  notPossibleIndividuals = [individual for individual in population if individual[2] > 0]

  # rank both lists
  if(ranked):
      
      # ranked by the fit function value
      possibleIndividuals = sorted(possibleIndividuals,key=itemgetter(1))
      
      # ranked by the number of infractions and by the sum of it
      notPossibleIndividuals = sorted(notPossibleIndividuals,key=itemgetter(2, 3))

  return  possibleIndividuals, notPossibleIndividuals


################################################################################
# Get Best                                                             PRIVATE #
################################################################################
def _getBest(population):
  '''
  Select the best individual of all the population
  
  input:
      
      population: list with the entire population
      
  output:
      
      returns the single best individual from the population
  '''
  
  # list and rank all the population with the _listIndividuals function
  rankedPossibles, rankedNotPossibles = _listIndividuals(population, ranked = True)
  
  # if the list of possibles are not empty then select its best
  if len(rankedPossibles) != 0:     return rankedPossibles[0]
  
  # if the list of possibles is empty select the best individual from the impossible ones
  else:     return rankedNotPossibles[0]
  
################################################################################
# Get Worst                                                            PRIVATE #
################################################################################
def _getWorst(population):
  '''
  Select the best individual of all the population
  
  input:
      
      population: list with the entire population
      
  output:
      
      returns the single worst individual from the population
  '''
  
  # list and rank all the population with the _listIndividuals function
  rankedPossibles, rankedNotPossibles = _listIndividuals(population, ranked = True)
  
  # if the list of impossibles are not empty then select its worst
  if len(rankedNotPossibles) != 0:      return rankedNotPossibles[-1]
  
  # if the list of impossibles is empty select the worst individual from the possible ones
  else:     return rankedPossibles[-1]


################################################################################
# Get Worst                                                             PUBLIC #
################################################################################
def GA(fit, lowerBoundaries, upperBoundaries, populationSize=50, generation=100, 
       crossover=0.5, mutation=0.01, **kwargs):
  '''
  Thats the main function for this library. It is where all the other functions
  are called and executed, therefore that's the function who shall be called into
  your code.
  
  It execute the Genetic Algorithm code with the possibility of adding constraints
  with arithmetic crossover and, as default, gaussian mutation.
  
  input:
      
      fit: fit function to be optimized. The function must receive a list and return
           a single value.
    
      lowerBoundarie: 1d-array with the minimum boundaries of the variables
      
      upperBoundarie: 1d-array with the maximum boundaries of the variables
      
      populationSize: the size of individuals in the population. State of the
                      art recomendation is from 20 to 50. Default is 50.
                    
      generation: number of generation to the training of the model. Default is 100.
      
      crossover: value between 0-1 to in wich the crossover will happen
      
      mutation: probability of mutation
      
      **kwargs:
          
          constraints: list with all the constraint functions.
          
          verbose: verbose = 1 will show all the details of each epoch. It is strongly
                   recomended to use verbose 0 if the numberTrainings is different then 1
          
          numberTrainings: number of times you want to repeat the optimization 
                           process
                           
          elitism: the number of parents who should be considered as possible parents
                   through elitism
      
  output:
      
      list containing the information from the best individual from each training
  '''
  
  print("################################################################")
  
  # kwargs input arguments input reading
  constraints, verbose, numberTrainings, elitism = [], 0, 1, 0
  for key,value in kwargs.items():
      if(key == 'constraints'): constraints = value
      if(key == 'verbose'): verbose = value
      if(key == 'numberTrainings'): numberTrainings = value
      if(key == 'elitism'): elitism = value
      
  
  
  bestIndividuals = []
  
  for j in range(numberTrainings):    
      # list of the best and worst individual for each interation   
      bestIndividual,worstIndividual = [],[]
      
      # initialization of the population
      population = _createPopulation(fit, lowerBoundaries, upperBoundaries, constraints, populationSize)
      
      # best individual from the initial population
      bestFound = _getBest(population)
      
      if(verbose == 0):
        
        toolbarWidht = 50
        
        oneProgress = int(generation/toolbarWidht)
    
      msgVerbose = 'Fitness: {:>.5E} - Number of Infractions: {:>d} - Sum of Infractions: {:.4f}'
      
      print('Training %d/%d'%(j+1, numberTrainings))
      
      # beggining of the iteration process
      for i in range(generation):
        
        #create elitism
        if (elitism is not 0): possibleParents = population[:elitism]
        
        else: possibleParents = population
        
        # creates a list of parents using the binary tournament
        parents = _binaryTournament(possibleParents)
        
        # create the population for the next generation
        population = _nextGeneration(parents, fit, constraints, lowerBoundaries, 
                                     upperBoundaries, crossover, mutation, verbose, populationSize = populationSize)
        
        # get the best individual from the new generation
        bestIndividual = _getBest(population)
        
        # get the best individual from the new generation
        worstIndividual = _getWorst(population) 
        
        # select the best individual from the generation
        bestFound = _selectIndividual(bestIndividual, bestFound)
    
        # write the progress bar
        if(verbose == 0 and i%oneProgress == 0):
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%% "\
                           "FitFunction = %.5f - Number of Infractions = %d - "\
                           "Sum of Infractions = %.2f" % ('='*int(i/oneProgress), 
                             (100*(i/generation)), bestFound[1],bestFound[2],
                             bestFound[3]))
            sys.stdout.flush()
            
        if(verbose == 1):
        
            print("################################################################\n")
            print("GENERATION (%d/%d)"%(i+1,generation))
            print("Best Individual:", bestIndividual[0])
            print(msgVerbose.format(bestIndividual[1],bestIndividual[2],bestIndividual[3]))
            print('---')
            print("\nWorst Individual:", worstIndividual[0])
            print(msgVerbose.format(worstIndividual[1],worstIndividual[2],worstIndividual[3]))
    
        
      # finish the progress bar
      if (verbose == 0):
          sys.stdout.write('\r')
          sys.stdout.write("[%-50s] %d%% "\
                           " - FitFunction = %.5f - Number of Infractions = %d - "\
                           "Sum of Infractions = %.2f"% ('='*int(toolbarWidht), (100),
                            bestFound[1],bestFound[2],bestFound[3])); sys.stdout.flush()
          sys.stdout.write('\n');sys.stdout.flush()
    
          print("Best Individual:", bestFound[0])
          print("################################################################")
      
      # append the best result from this training with all the others
      bestIndividuals.append(bestFound)
  
  return bestIndividuals
