from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS_Rollout import MCTS_Rollout
import torch
import time
import numpy as np

start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
end = time.time()
print('Time: ',end-start, ' s')
#device = 'cpu'
system_size =3
toric_code = Toric_code(system_size)
toric_code.generate_random_error(0.12)

args = {'cpuct': np.sqrt(2), 'num_simulations':100, 'grid_shift': system_size//2, 'disscount_factor':0.95, 'rollout_length':20}

start = time.time()
for i in range(1):
    mcts = MCTS_Rollout(device, args, toric_code)
    
    start = time.time()
    Qsa_max, all_Qsa = mcts.get_maxQsa()
    
    end = time.time()
    print('MCTS time: ',end-start, ' s')
    print('_________________')
    #print('antal perspektiv:', len(pi))
    print('all Qsa:', all_Qsa)
    print('max Qsa:', Qsa_max)
    #print('action:', action)
    print('_________________')

end = time.time()
print('Time: ',end-start, ' s')
