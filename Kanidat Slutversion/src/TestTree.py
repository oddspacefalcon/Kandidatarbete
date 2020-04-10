import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import time
from .TestTree2 import TestTree2
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
        self.Wsa = {}#Wsa         # stores total value policy for node
        self.Asav = {}#Asav      # stores (s,a,v) for that branch as key  and the action as value. 
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
        self.taken_actions = []
        self.states_to_leafnode = []
        self.actions_to_leafnode = []
        self.actions_to_leafnode_nostring = []
        self.reward = 0 #save reward when we go to leaf node
        self.TrivialLoop = False

    def get_Qvals(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)
        
        #.............................Search...............................
        perspective_list = self.generate_perspective(self.args['grid_shift'], copy.deepcopy(self.syndrom))
        for i in range(self.args['num_simulations']):
            if self.TrivialLoop:
               continue
            self.search(copy.deepcopy(self.toric_code), actions_taken, perspective_list)
            self.loop_check.clear()
            
            self.states_to_leafnode.clear()
            self.actions_to_leafnode.clear()
            self.actions_to_leafnode_nostring.clear()
        
       #..............................Max Qsa .............................        
        Ws_temp = {}
        action_temp = {}
        for key, value in self.Wsa.items():
            if s in key:
                Ws_temp[(key)] = value
                action_temp[key[1]] = value

        key = max(Ws_temp, key=Ws_temp.get)
        max_value = str(Ws_temp[(key)])
        best_action = self.Asav[str(key[0]), str(key[1]), max_value]
        maxQ = Ws_temp[(key)] 
        #del self.Asav[str(key[0]), str(key[1]), max_value]
        #del self.Wsa[(key)]
        actions = self.get_possible_actions(self.syndrom)
        perspectives = self.generate_perspective(self.args['grid_shift'], self.syndrom)
        perspectives = Perspective(*zip(*perspectives)).perspective

        all_Qsa = np.array([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])
        return maxQ, best_action, all_Qsa
        
        '''
        actions = self.get_possible_actions(self.syndrom)
        perspectives = self.generate_perspective(self.args['grid_shift'], self.syndrom)
        perspectives = Perspective(*zip(*perspectives)).perspective

        all_Qsa = np.array([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])
        
        return (all_Qsa, perspectives, actions)
        '''
    
    def get_qs_actions(self, state):
        s = str(state)
        Ws_temp = {}
        action_temp = {}
        for key, value in self.Wsa.items():
            if s in key:
                Ws_temp[(key)] = value
                action_temp[key[1]] = value
 
        print(self.Wsa)
        key = max(Ws_temp, key=Ws_temp.get)
        max_value = str(Ws_temp[(key)])
        best_action = self.Asav[str(key[0]), str(key[1]), max_value]
        maxQ = Ws_temp[(key)] 
       
 
        return best_action
    
    def search(self, toric_code, actions_taken, perspective_list):
        with torch.no_grad():
            s = str(toric_code.current_state)
            
            #...........................Check if terminal state.............................
            #if no trivial loop check if terminal state
            terminal_state = toric_code.terminal_state(toric_code.current_state)
            toric_code.eval_ground_state()
            if terminal_state == 0 and toric_code.ground_state == True:
                return 100
            elif terminal_state == 0 and toric_code.ground_state == False:
                return -100  
    
            # ............................Rollout and backprop......................................

            v = self.rollout(perspective_list, copy.deepcopy(toric_code), actions_taken) + self.reward 
            # If this state has not been visited.
            self.backpropagation(v) 

            return

    def backpropagation(self, v):
        i = 0
        for s in self.states_to_leafnode:
            a = self.actions_to_leafnode[i]
            if (s,a) in self.Qsa:
                self.Wsa[(s,a)] = self.Wsa[(s,a)] + v  
                self.Qsa[(s,a)] = self.Wsa[(s,a)]/self.Nsa[(s,a)]
                self.Nsa[(s,a)] += 1

                # to get best action later on...
                temp_v = str(self.Wsa[(s,a)])
                self.Asav[(s,a,temp_v)] = self.actions_to_leafnode_nostring[i]

            else:
                self.Wsa[(s,a)] = v
                self.Nsa[(s,a)] = 1

                temp_v = str(self.Wsa[(s,a)])
                self.Asav[(s,a,temp_v)] = self.actions_to_leafnode_nostring[i]
            
                self.Qsa[(s,a)] = self.Wsa[(s,a)]
            i += 1 
            self.Ns[s] += 1

    def rollout(self, perspective_list, toric_code, actions_taken):
        counter = 0 #number of steps in rollout
        accumulated_reward = 0 
        v = 0
        discount = self.args['discount_factor']
        state = copy.deepcopy(toric_code.current_state)
        while True:
            counter += 1
            #if no trivial loop check if terminal state
            terminal_state = toric_code.terminal_state(toric_code.current_state)
            toric_code.eval_ground_state()
            if terminal_state == 0 and toric_code.ground_state == True:
                v += 100
                break
            elif terminal_state == 0 and toric_code.ground_state == False:
                v += -100
                break
            
            # om ej terminal state
            if terminal_state == 1 and counter <= self.args['rollout_length']:
                perspective_list = self.generate_perspective(self.args['grid_shift'], toric_code.current_state)
                s = str(toric_code.current_state)
                rand_action = self.take_random_step(toric_code, actions_taken)
                a = str(rand_action)

                current_state = copy.deepcopy(toric_code.current_state)
                toric_code.step(rand_action)
                toric_code.current_state = toric_code.next_state 
                next_state = copy.deepcopy(toric_code.current_state)

                #get reward for step
                accumulated_reward += self.get_reward(next_state, current_state, toric_code)*discount**(counter)
                
                # discount factor because want to promote early high rewards
                v += accumulated_reward 
    
            #Break if terminal state
            terminal_state = toric_code.terminal_state(toric_code.current_state)
            toric_code.eval_ground_state()
            if terminal_state == 0 and toric_code.ground_state == True:
                v += 100
                break
            elif terminal_state == 0 and toric_code.ground_state == False:
                v += -100
                break

            if counter == self.args['rollout_length']:
                break
        return v
    
    def take_random_step(self, toric_code, actions_taken):
        state = copy.deepcopy(toric_code.current_state)
        perspective_list = self.generate_perspective(self.args['grid_shift'], toric_code.current_state)
        current_state = copy.deepcopy(state)

        # only action in qubitmatrix 1 if qubitmatrix all zeros in 0 
        if np.sum(current_state[0]) == 0:
            pos_matrix_1 = []
            for i in perspective_list:
                if i.position[0] == 1:
                    pos_matrix_1.append(i.position)
            perspective_index_rand = random.randint(0,len(pos_matrix_1)-1)
            rand_pos = pos_matrix_1[perspective_index_rand]
            action_index_rand = random.randint(1,3)
            rand_action = Action(np.array(rand_pos), action_index_rand) 
        
        elif np.sum(current_state[1]) == 0:     # only action in qubitmatrix 0 if qubitmatrix all zeros in 1           
            pos_matrix_0 = []
            #samlar perspektiven i matrix 0 i en array
            for i in perspective_list:
                if i.position[0] == 0:
                    pos_matrix_0.append(i.position)
            perspective_index_rand = random.randint(0,len(pos_matrix_0)-1)
            rand_pos = pos_matrix_0[perspective_index_rand] 
            action_index_rand = random.randint(1,3)  
            rand_action = Action(np.array(rand_pos), action_index_rand) 
        
        else: #get random action from both matrices
            perspective_index_rand = random.randint(0,len(perspective_list)-1)
            rand_pos = perspective_list[perspective_index_rand].position
            action_index_rand = random.randint(1,3)
            rand_action = Action(np.array(rand_pos), action_index_rand)

        return rand_action

    def best_index(self, Qvals):
        return np.unravel_index(np.argmax(Qvals), Qvals.shape)      
        
    def next_step(self, action):
        self.toric_code.step(action)
        self.syndrom = self.toric_code.next_state    
    
    def get_memory_Qvals(self, Qvals, perspectives, actions, nr_min_visits=5):
        #Takes in all the Qvalues and the corresponding perspectives and actions
        # and returns the ones that have been visited more than or equal to nr_min_visits
        new_Qval = []
        new_perspectives = []

        s = str(self.syndrom)

        for i, list_action in zip(range(len(Qvals)), actions):
            nr_visit_sum = 0
            for action in list_action:
                nr_visit_sum += self.Nsa[(s, str(action))] if (s,str(action)) in self.Nsa else 0
            if nr_visit_sum >= nr_min_visits:
                new_Qval.append(Qvals[i])
                new_perspectives.append(perspectives[i])
        return (new_Qval, new_perspectives)

    def get_reward(self, next_state, current_state, toric_code):
        #if no trivial loop check if terminal state
        terminal_state = toric_code.terminal_state(toric_code.current_state)
        toric_code.eval_ground_state()
        if terminal_state == 0 and toric_code.ground_state == True:
            reward = 100
        elif terminal_state == 0 and toric_code.ground_state == False:
            reward = -100
        else:
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
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