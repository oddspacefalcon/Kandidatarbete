from src.toric_model import Toric_code
from ResNet import ResNet18, ResNet152
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
from src.util import convert_from_np_to_tensor
import timeit
import numpy as np
from src.MCTS_vector import MCTS_vector
import time

def test_multiple():
    Ts_stack =  []
    sizes = [1, 5]
    for i in sizes:

        device = 'cpu'
        system_size = 5
        toric_codes = [Toric_code(system_size) for _ in range(i)]
        for toric_code in toric_codes:
            toric_code.generate_random_error(0.1)
        
        model = ResNet18()

        model.eval()
        args = {'cpuct': 5, 'num_simulations':50, 'grid_shift': system_size//2, 'disscount_factor':0.95}

        mcts = MCTS_vector(model, device, args, toric_codes)
        Qs, perspective_batch, action_batch = mcts.get_Qvals()

        perspective_index, action_index = mcts.get_best_indices(Qs)
        maxQs = [Qs[i][pi][ai] for i, pi, ai in zip(range(mcts.nr_trees), perspective_index, action_index)]
        print('-------------------------')
        print('Q:', maxQs)
        print('-------------------------')

        print("Time for different sections (per tree) : {}".format(mcts.Ts/i))
        print("How many times different sections run: {}".format(mcts.Nrs))
        print("Deepest level {}".format(mcts.max_level))
        print("Max loop iterations {}".format(mcts.max_loops))
        Ts_stack.append(mcts.Ts/i)

    print(sizes)
    print(np.array(Ts_stack))

def compare_vector():
    num_sim = [5, 10, 50]
    time_vector = []
    time_normal = []
    for nr in num_sim:
        model = ResNet18()

        device = 'cpu'
        system_size = 5
        toric_codes = [Toric_code(system_size) for _ in range(1)]
        for toric_code in toric_codes:
            toric_code.generate_random_error(0.1)

        model.eval()
        args = {'cpuct': 5, 'num_simulations':nr, 'grid_shift': system_size//2, 'disscount_factor':0.95}

        mcts_vec = MCTS_vector(model, device, args, toric_codes)
        t1 = time.clock()
        Qs, perspective_batch, action_batch = mcts_vec.get_Qvals()
        time_vector.append(t1-time.clock())

        mcts = MCTS(model, device, args, toric_codes[0])

        
        t1 = time.clock()
        q, a = mcts.get_qs_actions()
        time_normal.append(t1-time.clock())

    print("time for normal mcts:\n{0}\n---------------------------------------------------------------\ntime for vector mcts:\n{1}".format(np.array(time_normal), np.array(time_vector)))

compare_vector()