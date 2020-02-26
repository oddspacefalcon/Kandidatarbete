from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
import numpy as np
from src.util import convert_from_np_to_tensor

for i in range(3):
    device = 'cpu'
    system_size = 7
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18()
    args = {'cpuct': 50, 'num_simulations':30, 'grid_shift': system_size//2}

    mcts = MCTS(toric_code, model, device, args)
    pi, action = mcts.get_probs_action()

    print('pi:', pi)
    print('action:', action)
    print('----------------------------------')
