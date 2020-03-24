import numpy as np
from src.util import Action, Perspective, convert_from_np_to_tensor
from src.toric_model import Toric_code
import math
import numpy as np
import copy
from ResNet import ResNet18
import torch
import random

device = 'cpu'
system_size = 3
toric = Toric_code(system_size)
toric.generate_random_error(0.12)

Nsa = {}
Ns = {}
Ps = {}
Qsa = {}
nnet = ResNet18()

# root state s
root_state = np.copy(toric.qubit_matrix)
s = str(root_state)


# pi policy
#array_of_perspectives = toric.generate_perspective(system_size//2, toric.current_state) 
#perspective_pos = Perspective(*zip(*array_of_perspectives)).position
#actions = [[str(Action(p_pos, x)) for x in range(3)] for p_pos in perspective_pos]
#pi = [[Nsa[(s,a)] if (s,a) in Nsa else 0 for a in perspective] for perspective in actions]


 #slänger ihop alla perspektiv för ett state till en batch
perspective_list = toric.generate_perspective(system_size//2, toric.current_state)
number_of_perspectives = len(perspective_list)-1
perspectives = Perspective(*zip(*perspective_list))
batch_perspectives = np.array(perspectives.perspective)
batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
batch_perspectives = batch_perspectives.to(device)
batch_position_actions = perspectives.position

perspective_index_rand = random.randint(0,len(perspective_list)-1)
rand_pos = perspective_list[perspective_index_rand].position
action_index = random.randint(1,3)
rand_action = Action(np.array(rand_pos), action_index)

size = system_size
actions_taken = np.zeros((2,size,size), dtype=int)

print(np.log(5))

