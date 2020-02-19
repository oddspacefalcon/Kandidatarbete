import numpy as np
from toric_model import Toric_code
from util import Transition, Action, Perspective
from random import random
from MCTS import MCTS
from collections import namedtuple

#dummy klass som ändast används för att testa 
class NNet():

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def feedforward(self, input_perspective):
        return (random(), np.random.rand(input_perspective.shape[0], 3))
system_size = 3
toric = Toric_code(system_size)
toric.generate_n_random_errors(2)
Arguments = namedtuple('Arguments', ['nr_simulations', 'gridshift', 'cpuct'])
# args = dotdict({
#     'nr_simulations': 100,
#     'gridshift': 5//2,
#     'cpuct': 2 
# })
args = Arguments(10, system_size//2, 2)
nnet = NNet(system_size)

myMontyCarl = MCTS(args, nnet, toric)

print(myMontyCarl.get_actionprob())