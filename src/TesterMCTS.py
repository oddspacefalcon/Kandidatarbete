import numpy as np
from toric_model import Toric_code
from util import Transition, Action, Perspective
from random import random
from MCTS import MCTS
from collections import namedtuple
from ResNet import ResNet18

system_size = 3
toric = Toric_code(system_size)
toric.generate_random_error(0.1)
Arguments = namedtuple('Arguments', ['nr_simulations', 'gridshift', 'cpuct'])

args = Arguments(100, system_size//2, 50)
device = 'cpu'
nnet = ResNet18()


myMontyCarl = MCTS(args, nnet, toric, device)

pi, v, action = myMontyCarl.get_policy()

print('pi:', pi)
print('v:', v)
print('action', action)
print('-----------------------')