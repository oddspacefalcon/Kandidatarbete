from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS_rollout2 import MCTS_rollout2
import torch
import time
import numpy as np
import copy

device = 'cpu'
system_size =3
toric_code = Toric_code(system_size)
toric_code.generate_random_error(0.3)
args = {'cpuct': np.sqrt(2), 'num_simulations':300, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':1, \
'actions_to_explore':7*3}
Ns = {}
Nsa = {}
Qsa = {}
Wsa = {}

start = time.time()
mcts = MCTS_rollout2(device, args, Ns, Nsa, Qsa, Wsa, toric_code)
Qsa_max, all_Qsa, best_action, Qsa, Wsa, Nsa, Ns = mcts.get_maxQsa()
end = time.time()

def step(action, syndrom):
    qubit_matrix = action.position[0]
    row = action.position[1]
    col = action.position[2]
    add_opperator = action.action
    rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)

    #if x or y
    if add_opperator == 1 or add_opperator ==2:
        if qubit_matrix == 0:
            syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[0][row][(col-1)%system_size] = (syndrom[0][row][(col-1)%system_size]+1)%2
        elif qubit_matrix == 1:
            syndrom[1][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[1][(row+1)%system_size][col] = (syndrom[1][(row+1)%system_size][col]+1)%2
    #if z or y
    if add_opperator == 3 or add_opperator ==2:
        if qubit_matrix == 0:
            syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[0][(row-1)%system_size][col] = (syndrom[0][(row-1)%system_size][col]+1)%2
        elif qubit_matrix == 1:
            syndrom[1][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[1][row][(col+1)%system_size] = (syndrom[1][row][(col+1)%system_size]+1)%2


state = copy.deepcopy(toric_code.current_state)
size = system_size
actions_taken = np.zeros((2,size,size), dtype=int)
print('_____________________________')
print('_____________________________')
current_state = copy.deepcopy(state)
step(best_action, state)
next_state = copy.deepcopy(state)
print(current_state)
print('-----------')
print(best_action)
print(next_state)
add1 = np.sum(current_state)
add2 = np.sum(next_state)
print('sum diff: ', add1-add2)


counter = 0
while True:
    if counter % 5 == 0:
        #args = {'cpuct': np.sqrt(2), 'num_simulations':50, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':5, \
        #'actions_to_explore':7*3}
        Ns = {}
        Nsa = {}
        Qsa = {}
        Wsa = {}

    all_zeros = not np.any(next_state)
    if all_zeros:
        print('We Wooooon!!! ', next_state)
        break

    mcts = MCTS_rollout2(device, args, Ns, Nsa, Qsa, Wsa, None, next_state)
    Qsa_max, all_Qsa, best_action, Qsa, Wsa, Nsa, Ns = mcts.get_maxQsa()

    state = copy.deepcopy(next_state)
    size = system_size
    actions_taken = np.zeros((2,size,size), dtype=int)
    print('_____________________________')
    print('_____________________________')
    current_state = copy.deepcopy(state)
    step(best_action, state)
    next_state = copy.deepcopy(state)
    print(current_state)
    print('-----------')
    print(best_action)
    print(next_state)
    add1 = np.sum(current_state)
    add2 = np.sum(next_state)
    print('sum diff: ', add1-add2)





