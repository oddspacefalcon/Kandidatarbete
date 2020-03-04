import numpy as np
from src.toric_model import Toric_code
from src.util import Transition, Action, Perspective
from random import random
from src.MCTS2 import MCTS2
from collections import namedtuple
from ResNet import ResNet18

system_size = 5
toric = Toric_code(system_size)
toric.generate_random_error(0.1)
Arguments = namedtuple('Arguments', ['nr_simulations', 'gridshift', 'cpuct', 'disscount_factor'])

args = Arguments(20, system_size//2, 10, 0.95)
device = 'cpu'
nnet = ResNet18() #.to(device)


myMontyCarl = MCTS2(args, nnet, toric, device)

pi, v, action = myMontyCarl.get_policy()

print('--------------------------------------------------------------------')
print('pi:', pi)
print('v:', v)
print('action', action)
print('--------------------------------------------------------------------')