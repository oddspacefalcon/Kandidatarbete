
from src.toric_model import Toric_code
from src.toric_model import Action
from src.toric_model import Perspective
import numpy as np
import random


system_size = 5
grid_shift = system_size//2
p_error = 0.1

toric = Toric_code(system_size)

toric.generate_random_error(p_error)

perspectives = toric.generate_perspective(grid_shift, toric.current_state)
perspectives = Perspective(*zip(*perspectives))
actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
action = actions[2][2]
action2 = actions[2][1]
print(action)
print(action2)


equal = np.array_equal(action.position, action2.position)
if equal and action.action==action2.action:
    print('wut')




###################################
a = np.array([[200,300,400],[100,50,300]])

index2 = []
for i in range(len(a)):
    n = 0
    for j in a[i]:
        if j == 300:
            index2.append((i,n))
        n += 1
rand_index = index2[random.randint(0, len(index2)-1)]
actions[rand_index[0]][rand_index[1]]
