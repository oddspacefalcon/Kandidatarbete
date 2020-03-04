from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
import numpy as np
from src.util import convert_from_np_to_tensor
import time

steps = 5
start_time = time.time()
print('gpu synlig:', torch.cuda.is_available())
print(torch.cuda.memory_reserved(device='cuda') * 1e-6, 'Mb')
for i in range(steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    system_size = 7
    toric_code = Toric_code(system_size)
    toric_code.generate_random_error(0.1)

    model = ResNet18().to(device)
    print(torch.cuda.memory_reserved(device='cuda') * 1e-6, 'Mb')

    model.eval()
    args = {'cpuct': 50, 'num_simulations':100, 'grid_shift': system_size//2}

    mcts = MCTS(toric_code, model, device, args)
    pi, action = mcts.get_probs_action()

    print('pi:', pi)
    print('action:', action)
    print('----------------------------------')

print('time per step: {0:.3} s'.format((time.time() - start_time) / steps))