from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
import numpy as np
from src.util import convert_from_np_to_tensor

for i in range(1):
    device = 'cpu'
    system_size = 7
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18()
    args = {'cpuct': 0.5, 'num_simulations':30, 'grid_shift': system_size//2}

    mcts = MCTS(toric_code, model, device, args)
    pi, action = mcts.get_probs_action()

    print('antal perspektiv:', len(pi))
    print('pi:', pi)
    print('action:', action)
    print('----------------------------------')


# def alpha_loss(pi, z, p, v):
#     # (pi, z, p, v) ska vara torch tensorer
#     # L2 termen läggs till automatiskt om man sätter weight_decay=något tal, som argument när vi skapar optimizern
#      return torch.sum((z - v)**2) - torch.sum(pi.T * torch.log(p))
