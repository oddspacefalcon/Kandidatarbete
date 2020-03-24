from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
import torch
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system_size =3
toric_code = Toric_code(system_size)
toric_code.generate_random_error(0.1)

start = time.time()
model = ResNet18().to(device)
end = time.time()
print('Time model.to(cuda): ',end-start, ' s')

args = {'cpuct': 50, 'num_simulations':10, 'grid_shift': system_size//2, 'disscount_factor':0.95}

start = time.time()
for i in range(1):
    mcts = MCTS(model, device, args, toric_code)
    model.eval()
    
    
    start = time.time()
    Qsa_max, all_Qsa = mcts.get_qs_actions()
    
    end = time.time()
    print('MCTS time: ',end-start, ' s')
    
    #print('antal perspektiv:', len(pi))
    print('all_Qsa:', all_Qsa)
    print('Q:', Qsa_max)
    #print('action:', action)
    print('----------------------------------')

end = time.time()
print('Time: ',end-start, ' s')
