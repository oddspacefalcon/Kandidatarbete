from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
from src.util import convert_from_np_to_tensor

for i in range(1):

    device = 'cpu'
    system_size = 3
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18()
    args = {'cpuct': 5, 'num_simulations':10, 'grid_shift': system_size//2, 'disscount_factor':0.95}

    mcts = MCTS(model, device, args, toric_code)
    Qsa_max = mcts.get_probs_action()

    print('-------------------------')
    print('Q:', Qsa_max)
    print('-------------------------')