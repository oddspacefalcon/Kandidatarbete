import numpy as np
from src.util import Action, Perspective, convert_from_np_to_tensor
from src.toric_model import Toric_code
import math
import numpy as np
import copy
from ResNet import ResNet18
import torch

device = 'cpu'
system_size = 3
toric = Toric_code(system_size)
toric.generate_random_error(0.1)

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
array_of_perspectives = toric.generate_perspective(system_size//2, toric.current_state) 
number_of_perspectives = len(array_of_perspectives) - 1
perspectives = Perspective(*zip(*array_of_perspectives))
perspective_pos = perspectives.position
batch_perspectives = np.array(perspectives.perspective)
batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
batch_perspectives = batch_perspectives.to(device)
batch_position_actions = perspectives.position

current_level = 0 
actions = []

Ps[s] = nnet.forward(batch_perspectives)

pos = [1,0,2]
action = 1

best_action = Action(pos, action)
print(toric.current_state)
print('---------')
toric.step(best_action)
toric.current_state = toric.next_state
print(toric.next_state)



def step(self, action, syndrom, action_matrix):
        qubit_matrix = action.position[0]
        row = action.position[1]
        col = action.position[2]
        add_opperator = action.action
        rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)

        #Förändrar action matrisen
        current_state = action_matrix[qubit_matrix][row][col]
        action_matrix[qubit_matrix][row][col] = rule_table[add_opperator][current_state]

        #if x or y
        if add_opperator == 1 or add_opperator ==2:
            if qubit_matrix == 0:
                syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
                syndrom[0][row][(col-1)%self.system_size] = (syndrom[0][row][(col-1)%self.system_size]+1)%2
            elif qubit_matrix == 1:
                syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
                syndrom[0][(row+1)%self.system_size][col] = (syndrom[0][(row+1)%self.system_size][col]+1)%2
        #if z or y
        if add_opperator == 3 or add_opperator ==2:
            if qubit_matrix == 0:
                syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
                syndrom[0][(row-1)%self.system_size][col] = (syndrom[0][(row-1)%self.system_size][col]+1)%2
            elif qubit_matrix == 1:
                syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
                syndrom[0][row][(col+1)%self.system_size] = (syndrom[0][row][(col+1)%self.system_size]+1)%2







