from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS_vector import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
from src.util import convert_from_np_to_tensor
import timeit

print()

for i in [1, 3, 5, 10, 20, 50, 100]:

    device = 'cpu'
    system_size = 5
    toric_codes = [Toric_code(system_size) for _ in range(i)]
    for toric_code in toric_codes:
        toric_code.generate_random_error(0.1)


    model = ResNet18()
    model.eval()
    args = {'cpuct': 5, 'num_simulations':5, 'grid_shift': system_size//2, 'disscount_factor':0.95}

    mcts = MCTS(model, device, args, toric_codes)
    Qsa_max = mcts.get_probs_action()

    print('-------------------------')
    print('Q:', Qsa_max)
    print('-------------------------')

    print("Time for different sections : {}".format(mcts.Ts))
    print("How many times different sections run: {}".format(mcts.Nrs))
    print("Deepest level {}".format(mcts.max_level))
    print("Max loop iterations {}".format(mcts.max_loops))