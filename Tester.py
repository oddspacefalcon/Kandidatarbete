from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system_size =3
toric_code = Toric_code(system_size)
toric_code.generate_random_error(0.1)

start = time.time()
model = ResNet18().to(device)
end = time.time()
print('Time model.to(cuda): ',end-start, ' s')

args = {'cpuct': 50, 'num_simulations':50, 'grid_shift': system_size//2, 'disscount_factor':0.95}

start = time.time()
mcts = MCTS(model, device, args, toric_code)
model.eval()
Qsa_max, all_Qsa, best_action = mcts.get_qs_actions()
end = time.time()

'''
print('_________________')
print('all Qsa:', all_Qsa)
print('max Qsa:', Qsa_max)
print('best action', best_action)
print('_________________')
print('MCTS time: ',end-start, ' s')
'''

counter = 0
while True:

    all_zeros = not np.any(toric_code.current_state)
    if all_zeros:
        print('We Wooooon!!! ', toric_code.current_state)
        break

    mcts = MCTS(model, device, args, toric_code)
    model.eval()
    Qsa_max, all_Qsa, best_action = mcts.get_qs_actions()

    print(toric_code.current_state)
    print('---------------')
    print('best action', best_action)
    toric_code.step(best_action)
    toric_code.current_state = toric_code.next_state
    print(toric_code.current_state)
    print('_________________')
    counter +=1
    print('Num of searches: ',counter)
    print('Sum of errors: ',np.sum(toric_code.current_state))
    #print('all Qsa:', all_Qsa)
    print('______________________________________')



