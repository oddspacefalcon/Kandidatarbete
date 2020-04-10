from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS_Rollout import MCTS_Rollout
import torch
import time
import numpy as np
import copy


device = 'cpu'
system_size =3
num_sim = 1000
new_tree_after_move = 5
P_error = 0.3
rolloout = 4

toric_code = Toric_code(system_size)
toric_code.generate_random_error(P_error)
args = {'cpuct': np.sqrt(2), 'num_simulations':num_sim, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':rolloout}

Ns = {}
Nsa = {}
Qsa = {}
Wsa = {}
win = 0
loss = 0

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
            syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
            syndrom[1][(row+1)%system_size][col] = (syndrom[1][(row+1)%system_size][col]+1)%2
    #if z or y
    if add_opperator == 3 or add_opperator ==2:
        if qubit_matrix == 0:
            syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[0][(row-1)%system_size][col] = (syndrom[0][(row-1)%system_size][col]+1)%2
        elif qubit_matrix == 1:
            syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
            syndrom[1][row][(col+1)%system_size] = (syndrom[1][row][(col+1)%system_size]+1)%2

def firstMCTS(state):
    Ns = {}
    Nsa = {} 
    Qsa = {} 
    Wsa = {}
    start = time.time() 
    mcts = MCTS_Rollout(device, args, Ns, Nsa, Qsa, Wsa, toric_code, None)
    all_Qsa, all_actions, best_action = mcts.get_qs_actions()
    end = time.time()

    print(all_Qsa)
    
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
    print('total errors', add2)
    
    return next_state

def loopMCTS(state, max_moves, winn, losss):
    counter = 0
    win = 0
    loss = 0 
    while True:
        counter += 1
    
        all_zeros = not np.any(state)
        if all_zeros:
            print('We Wooooon!!! ', state)
            win = 1
            break
        if counter > (max_moves+7):
            print('To many moves..:(')
            loss = 1
            break

        Ns = {}
        Nsa = {} 
        Qsa = {} 
        Wsa = {}
        start = time.time() 
        mcts = MCTS_Rollout(device, args, Ns, Nsa, Qsa, Wsa, toric_code, None)
        all_Qsa, all_actions, best_action = mcts.get_qs_actions()
        end = time.time()

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
        print('total errors', add2)
        print('nr wins:', winn)
        print('nr loss: ', losss)
        print('MCTS time: ',end-start, ' s')
    
    return win, loss

'''
toric_code = Toric_code(system_size)
toric_code.generate_random_error(P_error)
try:
    next_state = firstMCTS(copy.deepcopy(toric_code.current_state))
    max_moves = np.sum(next_state)
    #win1, loss1 = loopMCTS(copy.deepcopy(next_state), max_moves, win, loss)        
except ValueError:
    print(':(')
'''


count = 0
for i in range(100):
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(P_error)
    try:
        next_state = firstMCTS(copy.deepcopy(toric_code.current_state))
        max_moves = np.sum(next_state)
        win1, loss1 = loopMCTS(copy.deepcopy(next_state), max_moves, win, loss)        
    except ValueError:
        continue

    win += win1
    loss += loss1
    print('nr wins:', win)
    print('nr loss: ', loss)
    print('win ration: ', win /(win + loss))

