from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS_Rollout import MCTS_Rollout
import torch
import time
import numpy as np

device = 'cpu'
system_size =3
toric_code = Toric_code(system_size)
toric_code.generate_random_error(0.1)
args = {'cpuct': np.sqrt(2), 'num_simulations':50, 'grid_shift': system_size//2, 'discount_factor':0.95, 'rollout_length':50, \
'actions_to_explore':7*3}

counter = 0
while True:
    
    all_zeros = not np.any(toric_code.current_state)
    if all_zeros:
        print('We Wooooon!!! ', toric_code.current_state)
        break

    mcts = MCTS_Rollout(device, args, toric_code)
    Qsa_max, all_Qsa, best_action = mcts.get_maxQsa()

    print(toric_code.current_state)
    print('---------------')
    print('best action', best_action)
    toric_code.step(best_action)
    toric_code.current_state = toric_code.next_state
    print(toric_code.current_state)
    print('_________________')
    counter +=1
    print('Num of searches: ',counter)
    print(np.sum(toric_code.current_state))
    print('all Qsa:', all_Qsa)

    print('______________________________________')

    '''
def mcts_search():
    start = time.time()
    mcts = MCTS_Rollout(device, args, toric_code)
    Qsa_max, all_Qsa, best_action = mcts.get_maxQsa()
    end = time.time()
    
    print('_________________')
    print('all Qsa:', all_Qsa)
    print('max Qsa:', Qsa_max)
    print('best action', best_action)
    print('_________________')
    print('MCTS time: ',end-start, ' s')
'''