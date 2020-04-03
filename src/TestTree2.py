import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import time
EPS = 1e-8

class TestTree():

    def __init__(self, device, args, toric_code=None, syndrom=None):

        self.toric_code = toric_code # toric_model object

        if toric_code is None:
            if syndrom is None:
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
        else:
            self.syndrom = self.toric_code.current_state

        self.args = args     # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = {}      # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}        # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Wsa = {}         # stores total value policy for node
        self.Actions_s = {}
        self.device = device # 'cpu' or 'cuda'
        self.actions = []
        self.current_level = 0
        self.loop_check = set() # set that stores state action pairs
        self.system_size = self.syndrom.shape[1]
        self.next_state = []
        self.current_state = []
        self.atLeafNode = False

        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.ground_state = True    # True: only trivial loops, 
                                   # False: non trivial loop 
        self.states_to_leafnode = []
        self.taken_actions = []
   

    def get_maxQsa(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)
        

        
        #..................Get perspectives and batch for network......................
        perspective_list = self.generate_perspective(self.args['grid_shift'], copy.deepcopy(self.syndrom))
        number_of_perspectives = len(perspective_list)
        perspectives = Perspective(*zip(*perspective_list))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        #print('perspective list ', perspective_list)
        #.............................Search...............................

        for i in range(self.args['num_simulations']):
            self.search(copy.deepcopy(self.syndrom), actions_taken, perspective_list)
            self.loop_check.clear()
                    
       #..............................Max Qsa .............................

        actions = self.get_possible_actions(self.syndrom)
        all_Qsa2D = np.array([[self.Wsa[str(a)] if (str(a)) in self.Wsa else 0 for a in position] for position in actions])
        all_Qsa = np.reshape(all_Qsa2D, all_Qsa2D.size)
        maxQ = (all_Qsa != 0).argmax()
               
        #print(max(self.Wsa, key=self.Wsa.get))
        key = max(self.Wsa, key=self.Wsa.get)
        #print(self.Qsa[key, (str(self.Wsa[key]))])
        best_action = self.Qsa[key, (str(self.Wsa[key]))]
        maxQ = self.Wsa[key]
        all_Qsa = self.Wsa

        #sprint('Wsa', self.Wsa)
           
        return maxQ, all_Qsa, best_action


    def search(self, state, actions_taken, perspective_list):
        with torch.no_grad():
            s = str(state)
            
            #...........................Check if terminal state.............................
            '''
            #Check if non trivial loop
            self.qubit_matrix = state
            self.eval_ground_state()
            if self.ground_state is False:
                v = -100
                print('trivial loop :(')
                return v
            '''

            #if no trivial loop check if terminal state
            all_zeros = not np.any(state)
            if all_zeros:   
                #Trivial loop --> gamestate won!
                v = 100
                return v

            #..................rollout and backpropagation...................................
            v, a = self.rollout(perspective_list, copy.deepcopy(state), actions_taken)
            self.backpropagation(v, a)
            return


    def backpropagation(self, v, action):
        a = str(action)
        if (a) in self.Wsa:
            self.Wsa[a] = self.Wsa[a] + v 
            temp = str(self.Wsa[a])
            self.Qsa[(a,temp)] = action
            self.Nsa[a] += 1
            if a not in self.taken_actions:
                self.taken_actions.append(a)
            
        else:
            self.Wsa[a] = v
            temp = str(self.Wsa[a])
            self.Qsa[(a,temp)] = action
            self.Nsa[a] = 1
        
            
    def rollout(self, perspective_list, state, actions_taken):
        counter = 0 #number of steps in rollout
        accumulated_reward = 0 
        v = 0

        #.................take random action.............................
        #get random action
        perspective_index_rand = random.randint(0,len(perspective_list)-1)
        rand_pos = perspective_list[perspective_index_rand].position
        action_index_rand = random.randint(1,3)
        rand_action = Action(np.array(rand_pos), action_index_rand)
        #take random step
        current_state = copy.deepcopy(state)
        self.step(rand_action, state, actions_taken)
        next_state = copy.deepcopy(state)
        #print(rand_action)

    
        all_zeros = not np.any(state)
        #Check if non trivial loop
        self.qubit_matrix = state
        self.eval_ground_state()
        if self.ground_state is False:
            v = -100
            #print('trivial loop')
            
        #get reward for step
        accumulated_reward = self.get_reward(next_state, current_state)

        v = accumulated_reward 

        #Break if terminal state
        all_zeros = not np.any(state)
        if all_zeros:
            v = 100 
 
        return v, rand_action
        
    # Reward
    def get_reward(self, next_state, current_state):
        terminal = np.all(next_state==0)
        if terminal == True:
            reward = 100
 
        else:
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
            if reward == 0:
                reward = -2

        return reward

    def generate_perspective(self, grid_shift, state):
        def mod(index, shift):
            index = (index + shift) % self.system_size 
            return index
        perspectives = []
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        # qubit matrix 0
        for i in range(self.system_size):
            for j in range(self.system_size):
                if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
                plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    temp = Perspective(new_state, (0,i,j))
                    perspectives.append(temp)
        # qubit matrix 1
        for i in range(self.system_size):
            for j in range(self.system_size):
                if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
                plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    new_state = self.rotate_state(new_state) # rotate perspective clock wise
                    temp = Perspective(new_state, (1,i,j))
                    perspectives.append(temp)
        
        return perspectives

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
                syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
                syndrom[1][(row+1)%self.system_size][col] = (syndrom[1][(row+1)%self.system_size][col]+1)%2
        #if z or y
        if add_opperator == 3 or add_opperator ==2:
            if qubit_matrix == 0:
                syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
                syndrom[0][(row-1)%self.system_size][col] = (syndrom[0][(row-1)%self.system_size][col]+1)%2
            elif qubit_matrix == 1:
                syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
                syndrom[1][row][(col+1)%self.system_size] = (syndrom[1][row][(col+1)%self.system_size]+1)%2
    
    def rotate_state(self, state):
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        rot_plaquette_matrix = np.rot90(plaquette_matrix)
        rot_vertex_matrix = np.rot90(vertex_matrix)
        rot_vertex_matrix = np.roll(rot_vertex_matrix, 1, axis=0)
        rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
        return rot_state
    
    def mult_actions(self, action_matrix1, action_matrix2):
        rule_table = np.array(([[0,1,2,3], [1,0,3,2], [2,3,0,1], [3,2,1,0]]), dtype=int)

        return [[[rule_table[qu1][qu2] for qu1, qu2 in zip(row1, row2)] for row1, row2 in zip(qu_mat1, qu_mat2)] for qu_mat1, qu_mat2 in zip(action_matrix1, action_matrix2)]
    
    def get_possible_actions(self, state):
        perspectives = self.generate_perspective(self.args['grid_shift'], state)
        perspectives = Perspective(*zip(*perspectives))
        return [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position] #bytte ut np.array(p_pos)

    def eval_ground_state(self):    # True: trivial loop
                                    # False: non trivial loop
	       # can only distinguish non trivial and trivial loop. Categorization what kind of non trivial loop does not work 
            # function works only for odd grid dimensions! 3x3, 5x5, 7x7        
        def split_qubit_matrix_in_x_and_z():
        # loops vertex space qubit matrix 0
            z_matrix_0 = self.qubit_matrix[0,:,:]        
            y_errors = (z_matrix_0 == 2).astype(int)
            z_errors = (z_matrix_0 == 3).astype(int)
            z_matrix_0 = y_errors + z_errors 
            # loops vertex space qubit matrix 1
            z_matrix_1 = self.qubit_matrix[1,:,:]        
            y_errors = (z_matrix_1 == 2).astype(int)
            z_errors = (z_matrix_1 == 3).astype(int)
            z_matrix_1 = y_errors + z_errors
            # loops plaquette space qubit matrix 0
            x_matrix_0 = self.qubit_matrix[0,:,:]        
            x_errors = (x_matrix_0 == 1).astype(int)
            y_errors = (x_matrix_0 == 2).astype(int)
            x_matrix_0 = x_errors + y_errors 
            # loops plaquette space qubit matrix 1
            x_matrix_1 = self.qubit_matrix[1,:,:]        
            x_errors = (x_matrix_1 == 1).astype(int)
            y_errors = (x_matrix_1 == 2).astype(int)
            x_matrix_1 = x_errors + y_errors

            return x_matrix_0, x_matrix_1, z_matrix_0, z_matrix_1

        x_matrix_0, x_matrix_1, z_matrix_0, z_matrix_1 = split_qubit_matrix_in_x_and_z()
        
        loops_0 = np.sum(np.sum(x_matrix_0, axis=0))
        loops_1 = np.sum(np.sum(x_matrix_1, axis=0))
        
        loops_2 = np.sum(np.sum(z_matrix_0, axis=0))
        loops_3 = np.sum(np.sum(z_matrix_1, axis=0))

        if loops_0%2 == 1 or loops_1%2 == 1:
            self.ground_state = False
        elif loops_2%2 == 1 or loops_3%2 == 1:
            self.ground_state = False