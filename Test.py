from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
import numpy as np
from src.util import convert_from_np_to_tensor
import time
import copy
import random


t = torch.tensor([1, 1, 2], device='cuda')



batch_size = 5

t2 = torch.zeros((batch_size, *t.shape), device='cuda')


for i in range(batch_size):
    t2[i] = t

print(t2.shape)