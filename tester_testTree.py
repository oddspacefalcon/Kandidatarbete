from src.toric_model import Toric_code
from ResNet import ResNet18
from src.TestTree import TestTree
import torch
import time
import numpy as np
import copy

# d = 5 och P_error= 0.25 ok, num_sim_long = 50, num_sim_short = 50, new_tree_after_move = 3 ---> win 98, loss 2 ---> Win ration 0.98
# d = 5 och P_error= 0.05 ok, num_sim_long = 300, num_sim_short = 300, new_tree_after_move = 1 ---> win 79, loss 12 ---> Win ration 0.90
# d = 5 och P_error= 0.1 ok, num_sim_long = 300, num_sim_short = 300, new_tree_after_move = 1 ---> win 84, loss 16 ---> Win ration 0.84


device = 'cpu'
system_size =3
num_sim_long = 50
num_sim_short = 50
new_tree_after_move = 5
P_error = 0.05

toric_code = Toric_code(system_size)
toric_code.generate_random_error(P_error)
win = 0
loss = 0
Asav ={}
Wsa = {}

args = {'cpuct': 1*np.sqrt(2), 'num_simulations':num_sim_long, 'grid_shift': system_size//2, 'discount_factor':0.5, 'rollout_length':4, \
'actions_to_explore':7*3}

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
    Asav = {}
    Wsa = {}
    start = time.time() 
    mcts = TestTree(device, args, Asav, Wsa, toric_code, None)
    Qsa_max, best_action, Asav, Wsa = mcts.get_maxQsa()
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
    
    
    return next_state
 
def loopMCTS(state, max_moves, winn, losss, Asav, Wsa):
    counter = 0
    win = 0
    loss = 0 
    while True:
        counter += 1
        args = {'cpuct': 50*np.sqrt(2), 'num_simulations':num_sim_short, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':2, \
            'actions_to_explore':7*3}
    
        all_zeros = not np.any(state)
        if all_zeros:
            print('We Wooooon!!! ', state)
            win = 1
            break
        if counter > (max_moves+7):
            print('To many moves..:(')
            loss = 1
            break

        if counter % new_tree_after_move == 0:
            args = {'cpuct': 50*np.sqrt(2), 'num_simulations':num_sim_long, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':4, \
                'actions_to_explore':7*3}
            Asav = {}
            Wsa = {}
        #Asav = {}
        #Wsa = {}
        start = time.time()
        mcts = TestTree(device, args, Asav, Wsa, None, state)
        Qsa_max, best_action, Asav, Wsa = mcts.get_maxQsa()
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


#next_state = firstMCTS(copy.deepcopy(toric_code.current_state))
#win1, loss1 = loopMCTS(copy.deepcopy(next_state))




count = 0
for i in range(100):
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(P_error)
    try:
        next_state = firstMCTS(copy.deepcopy(toric_code.current_state))
        max_moves = np.sum(next_state)
        win1, loss1 = loopMCTS(copy.deepcopy(next_state), max_moves, win, loss, Asav, Wsa)        
    except ValueError:
        continue

    win += win1
    loss += loss1
    print('nr wins:', win)
    print('nr loss: ', loss)
    print('win ration: ', win /(win + loss))
    
