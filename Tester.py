from src.toric_model import Toric_code
from ResNet import ResNet18, ResNet152
from src.MCTS import MCTS
#from src.MCTS2 import MCTS2
from src.util import Perspective, Action
import torch
import torch.nn as nn
from src.util import convert_from_np_to_tensor
import timeit
import numpy as np
import time

# def mcts_test():
#     for i in range(1):

#         device = 'cpu'
#         system_size = 5
#         toric_code = Toric_code(system_size)
#         toric_code.generate_random_error(0.1)

#         model = ResNet152()
#         args = {'cpuct': 1, 'num_simulations':10, 'grid_shift': system_size//2, 'discount_factor':0.95}

#         mcts = MCTS(model, device, args, toric_code)
#         mcts2 = MCTS2(model, device, args, toric_code)
#         t0 = time.process_time()
#         Qsa_max = mcts.get_Qvals()
#         print("total time mcts1: {}".format(time.process_time()-t0))
#         t0 = time.process_time()
#         Qsa_max = mcts2.get_Qvals()
#         print("total time mcts2: {}".format(time.process_time()-t0))


#         print('-------------------------')
#         print('Ts2: {}'.format(mcts2.Ts))
#         print('-------------------------')
#         print('Nrs2: {}'.format(mcts2.Nrs))
#         print('-------------------------')
#         print('visitmodel2: {}'.format(mcts2.visit_model))
#         print('-------------------------')



        #print(mcts.Ts[3]*mcts.Nrs[3]/mcts.Nrs[2])
def tester():
    device = 'cpu'
    system_size = 3
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18()
    args = {'cpuct': 1, 'num_simulations': 50, 'grid_shift': system_size//2, 'discount_factor':0.95}

    mcts = MCTS(model, device, args, toric_code)

    Qsa_max, _, _ = mcts.get_Qvals() 

tester()
#mcts_test()

def generate_perspective_time_tester():
    setup = '''from src.toric_model import Toric_code
toric = Toric_code(5)
toric.generate_random_error(0.2)
    '''
    statement='perspectives = toric.generate_perspective(5//2, toric.current_state)'
    time = timeit.timeit(stmt=statement, setup=setup, number=1000)
    print("total time: {}".format(time/1000))

def perspective_tester():
    setup = '''from src.toric_model import Toric_code
import numpy as np
from src.util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
toric = Toric_code(5)
toric.generate_random_error(0.2)'''
    statement = '''perspective_list = toric.generate_perspective(5//2, toric.current_state)
number_of_perspectives = len(perspective_list)
perspectives = Perspective(*zip(*perspective_list))
batch_perspectives = np.array(perspectives.perspective)
batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
batch_perspectives = batch_perspectives.to("cpu")
batch_position_actions = perspectives.position'''
    time = timeit.timeit(stmt=statement, setup=setup, number=1000)
    print("total time: {}".format(time/1000))


def qvals_time_tester():
    setup = '''from src.toric_model import Toric_code
from src.util import Action
import numpy as np
toric = Toric_code(5)
toric.generate_random_error(0.2)
s =str(toric.current_state)
perspectives = toric.generate_perspective(5//2, toric.current_state)
perspectives, positions = zip(*perspectives)
actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in positions]
Qsa = {}
for perspective, action in zip(perspectives, actions):
    for a in action:
        Qsa[(str(perspective), str(a))] = 0
for i in range(10):
    toric = Toric_code(5)
    toric.generate_random_error(0.2)
    s =str(toric.current_state)
    perspectives = toric.generate_perspective(5//2, toric.current_state)
    perspectives, positions = zip(*perspectives)
    actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in positions]
    for perspective, action in zip(perspectives, actions):
        for a in action:
            Qsa[(str(perspective), str(a))] = 0
    '''
    statement = 'current_Qsa = np.array([[Qsa[(s,str(a))] if (s, str(a)) in Qsa else 0 for a in opperator_actions] for opperator_actions in actions])'
    time = timeit.timeit(stmt=statement, setup=setup, number=1000)
    print("total time: {}".format(time/1000))

def qvals_time_tester2():
    setup = '''from src.toric_model import Toric_code
from copy import deepcopy
from src.util import Action
import numpy as np
toric = Toric_code(5)
toric.generate_random_error(0.2)
s =str(toric.current_state)
perspectives = toric.generate_perspective(5//2, toric.current_state)
perspectives, positions = zip(*perspectives)
nr_perspectives = len(perspectives)
Qsa = {}
Qsa[s] = np.zeros((nr_perspectives, 3))
for i in range(10):
    toric = Toric_code(5)
    toric.generate_random_error(0.2)
    s =str(toric.current_state)
    perspectives = toric.generate_perspective(5//2, toric.current_state)
    perspectives, positions = zip(*perspectives)
    nr_perspectives = len(perspectives)
    for perspective in perspectives:
        Qsa[str(perspective)]  = np.zeros((nr_perspectives, 3))
    '''
    statement = 'current_Qsa = Qsa[s] if s in Qsa else np.zeros((nr_perspectives, 3))'
    time = timeit.timeit(stmt=statement, setup=setup, number=1000)
    print("total time: {}".format(time/1000))

def UCBpuct(probability_matrix, actions, s, Qsa, Nsa, Ns):

    current_Qsa = np.array([[Qsa[(s,str(a))] if (s, str(a)) in Qsa else 0 for a in opperator_actions] for opperator_actions in actions])
    current_Nsa = np.array([[Nsa[(s,str(a))] if (s, str(a)) in Nsa else 0 for a in opperator_actions] for opperator_actions in actions])
    if s not in Ns:
        current_Ns = 0.001
    else:
        if Ns[s] == 0:
            current_Ns = 0.001
        else:
            current_Ns = Ns[s]
    #använd max Q-värde: (eller använda )
    return current_Qsa + 15*probability_matrix*np.sqrt(current_Ns/(1+current_Nsa))

def UCBpuct2(prob_matrix, Ns, Nsa, Qsa, s):
    current_Qsa = Qsa[s] if s in Qsa else np.zeros(prob_matrix.shape)
    current_Nsa = Nsa[s] if s in Nsa else np.zeros(prob_matrix.shape)
    if s not in Ns:
        current_Ns = 0.001
    else:
        if Ns[s] == 0:
            current_Ns = 0.001
        else:
            current_Ns = Ns[s]
    #använd max Q-värde: (eller använda )
    return current_Qsa + 15*prob_matrix*np.sqrt(current_Ns/(1+current_Nsa))

def return_parameters():
    Qsa = {}
    Nsa = {}
    Ns = {}

    for i in range(10):
        toric = Toric_code(5)
        toric.generate_random_error(0.2)
        s =str(toric.current_state)
        perspectives = toric.generate_perspective(5//2, toric.current_state)
        perspectives, positions = zip(*perspectives)
        nr_perspectives = len(perspectives)
        for perspective in perspectives:
            Qsa[str(perspective)] = np.zeros((nr_perspectives, 3))
            Nsa[str(perspective)] = np.zeros((nr_perspectives, 3))
        prob_mat = np.random.randn(nr_perspectives, 3)
    return (prob_mat, Ns, Nsa, Qsa, s)

def return_parameters2():
    Qsa = {}
    Nsa = {}
    Ns = {}

    for i in range(10):
        toric = Toric_code(5)
        toric.generate_random_error(0.2)
        s =str(toric.current_state)
        perspectives = toric.generate_perspective(5//2, toric.current_state)
        perspectives, positions = zip(*perspectives)
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in positions]
        nr_perspectives = len(perspectives)
        for perspective in perspectives:
            Qsa[str(perspective), str(actions)] = np.zeros((nr_perspectives, 3))
            Nsa[str(perspective), str(actions)] = np.zeros((nr_perspectives, 3))
        prob_mat = np.random.randn(nr_perspectives, 3)
    return (prob_mat, actions, s, Qsa, Nsa, Ns)

def testUCB2():
    setup = '''from src.toric_model import Toric_code
from copy import deepcopy
from src.util import Action
from __main__ import return_parameters2, UCBpuct
import numpy as np
toric = Toric_code(5)
toric.generate_random_error(0.2)
param = return_parameters2()
    '''
    statement = 'UCB = UCBpuct(*param)'
    time = timeit.timeit(stmt=statement, setup=setup, number=1000)
    print("total time: {}".format(time/1000))

def toric_tester():
    device = 'cpu'
    system_size = 3
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18()
    args = {'cpuct': 5, 'num_simulations':10, 'grid_shift': system_size//2, 'discount_factor':0.95}

    mcts = MCTS(model, device, args, syndrom=toric_code.current_state)
    Qsa_max = mcts.get_Qvals()

    print('-------------------------')
    print('Q:', Qsa_max)
    print('-------------------------')

# def generate_perspective2(self, grid_shift, state):
#     def mod(index, shift):
#         index = (index + shift) % self.system_size 
#         return index
#     perspectives = []
#     vertex_matrix = state[0,:,:]
#     plaquette_matrix = state[1,:,:]
#     # qubit matrix 0
#     for i in range(self.system_size):
#         for j in range(self.system_size):
#             if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
#             plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
#                 new_state = np.roll(state, grid_shift-i, axis=1)
#                 new_state = np.roll(new_state, grid_shift-j, axis=2)
#                 temp = Perspective(new_state, (0,i,j))
#                 perspectives.append(temp)
#     # qubit matrix 1
#     for i in range(self.system_size):
#         for j in range(self.system_size):
#             if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
#             plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
#                 new_state = np.roll(state, grid_shift-i, axis=1)
#                 new_state = np.roll(new_state, grid_shift-j, axis=2)
#                 new_state = self.rotate_state(new_state) # rotate perspective clock wise
#                 temp = Perspective(new_state, (1,i,j))
#                 perspectives.append(temp)
    
#     return perspectives

# qvals_time_tester()
# testUCB2()
#generate_perspective_time_tester()

def list_generator_tester():
    statement = '''
actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in positions]
    '''
perspective_tester()