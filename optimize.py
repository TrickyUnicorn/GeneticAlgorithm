from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection, ExponentialRankingSelection, LinearRankingSelection, RouletteWheelSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation
import numpy as np

from math import sin, cos

# Funkcja Belea
# Oczekiwana wartosc f(3,0.5) =0

# Function boundaries
XB = (-5, 5)
YB = (-5, 5)
# Decrete precision of binary sequence
EPS = 0.0001

# Population size
POPULATION_SIZE = 200

# Selection method
SELECTION = RouletteWheelSelection()

# Crossover method
# pc: probability of crossover (0,1)
PC = 0.8
# pe: probability of gene exchange (0,1)
PE = 0.5

# Mutation
# pm: probability of mutation (0, 1)
PM = 0.05
# pbm: probability of big mutation (5 times bigger than pbm)
PBM = PM * 5
# alpha: intensive factor (0.5, 1)
ALPHA = 0.6

# Run
# ng: Evolution iteration steps (generations number)
NG = 150


# Built-in best fitness analysis.

from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Define population.


indv_template = BinaryIndividual(ranges=[XB, YB], eps=EPS)
population = Population(indv_template=indv_template, size=POPULATION_SIZE).init()

# Create genetic operators.
selection = SELECTION
crossover = UniformCrossover(pc=PC, pe=PE)
mutation = FlipBitBigMutation(pm=PM, pbm=PBM, alpha=ALPHA)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])


# Define fitness function.
@engine.fitness_register
@engine.minimize

def fitness(indv):
    x,y = indv.solution
    return (1.5 - x + x*y)*(1.5 - x + x*y)  +  (2.25 - x + x*y*y)*(2.25 - x + x*y*y) + (2.625 -x + x*y*y*y)*(2.625 -x + x*y*y*y)


engine.run(ng=NG)
