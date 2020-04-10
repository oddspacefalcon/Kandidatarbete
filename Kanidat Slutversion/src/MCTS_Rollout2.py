import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import time
EPS = 1e-8

class MCTS_Rollout2():

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
        self.Asav = {}      # stores (s,a,v) for that branch as key  and the action as value. 
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
        self.targetQsa = {}
        self.taken_actions = []
        self.states_to_leafnode = []
        self.actions_to_leafnode = []
        self.actions_to_leafnode_nostring = []
        self.reward = 0 #save reward when we go to leaf node
        self.counter = 0

    def get_qs_actions(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)
        
        #.............................Search...............................

        for i in range(self.args['num_simulations']):
            self.counter = i + 1
            self.search(copy.deepcopy(self.syndrom), actions_taken, str(copy.deepcopy(self.syndrom)))
            self.loop_check.clear()

            self.states_to_leafnode.clear()
            self.actions_to_leafnode.clear()
            self.actions_to_leafnode_nostring.clear()
                    
       #..............................Max Qsa .............................
        print(self.Ns[s])
        actions = self.get_possible_actions(self.syndrom)
        temp = 0
        tot = 0 
        for i in range(len(actions)):
            for j in range(3):
                if (s,str(actions[i][j])) in self.Nsa:
                    print('Nsa:', self.Nsa[(s,str(actions[i][j]))], 'Qsa:', self.Qsa[(s,str(actions[i][j]))], ' Action: ', str(actions[i][j]))
                    if temp <= self.Nsa[(s,str(actions[i][j]))]:
                        temp = self.Nsa[(s,str(actions[i][j]))]
                        best_action = actions[i][j]
                        tot += self.Nsa[(s,str(actions[i][j]))]
                else:
                     print('Nsa:',0)

        print('Tot Nsa', tot)

               #..............................Max Qsa .............................
        actions = self.get_possible_actions(self.syndrom)
        all_Qsa = torch.tensor([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])
        all_targetQsa = torch.tensor([[self.targetQsa[(s,str(a))] if (s,str(a)) in self.targetQsa else 0 for a in position] for position in actions])

        #best action
        index_max = np.unravel_index((all_targetQsa).argmax(), all_targetQsa.shape)
        best_action = actions[index_max[0]][index_max[1]]

        return best_action


    def search(self, state, actions_taken, root_state):
        with torch.no_grad():
            s = str(state)
            
            #...........................Check if terminal state.............................
            #if no trivial loop check if terminal state
            all_zeros = not np.any(state)
            self.qubit_matrix = state
            self.eval_ground_state()
            if self.ground_state is False:
                return -100
            elif all_zeros:
                return 100
            
            #............................Get perspectives ......................
            perspective_list = self.generate_perspective(self.args['grid_shift'], copy.deepcopy(self.syndrom))
            perspectives = Perspective(*zip(*perspective_list))
           
            # ............................If leaf node......................................
            if s not in self.Ns:
                self.Ns[s] = 0
                v = self.rollout(perspective_list, copy.deepcopy(state), actions_taken) + self.reward 
                if s != root_state:
                    self.backpropagation(v)
                else:
                    self.Ns[s] = 1
                return
            else:
                self.reward = 0
                self.selection(s, state, perspectives, perspective_list, actions_taken, root_state)
           
            return

    def selection(self, s, state, perspectives, perspective_list, actions_taken, root_state):
        self.states_to_leafnode.append(s)
        # ..........................Get best action with UCB1...................................
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
        current_Qsa = np.array([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions])
        current_Nsa = np.array([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions])
        if s not in self.Ns:
            current_Ns = 0.00001
        else:
            if self.Ns[s] == 0:
                current_Ns = 1
            else:
                current_Ns = self.Ns[s]
        UpperConfidence = current_Qsa + self.args['cpuct']*np.sqrt(np.log(current_Ns)/(current_Nsa+EPS))

        #Choose action with higest UCB which has not been explored before
        while True:
            perspective_index, action_index = np.unravel_index(np.argmax(UpperConfidence), UpperConfidence.shape)
            best_perspective = perspective_list[perspective_index]

            action = Action(np.array(best_perspective.position), action_index+1)

            a = str(action)
            if((s,a) not in self.loop_check):
                self.loop_check.add((s,a))
                break
            else:
                UpperConfidence[perspective_index][action_index] = -float('inf')
        
        unique, counts = np.unique(UpperConfidence, return_counts=True)
        unique_dict = dict(zip(unique, counts))
        length_unique_dict = len(unique_dict)
        if (len(UpperConfidence[0])*len(UpperConfidence)) != length_unique_dict:
            action = self.take_random_step(copy.deepcopy(state), actions_taken)

        #...........................Go down the tree with best action...........................
        current_state = copy.deepcopy(state)
        self.step(action, state, actions_taken)
        next_state = copy.deepcopy(state)
        reward = self.get_reward(next_state, current_state)
        self.reward = reward
        self.actions_to_leafnode.append(a)
        self.actions_to_leafnode_nostring.append(action)
     
        self.search(state, actions_taken,root_state)
   

    def backpropagation(self, v):
        i = 0
        for s in self.states_to_leafnode:
            a = self.actions_to_leafnode[i]
            if (s,a) in self.Qsa:
                self.Wsa[(s,a)] = self.Wsa[(s,a)] + v  
                self.Qsa[(s,a)] = self.Wsa[(s,a)]/self.Nsa[(s,a)]
                self.Nsa[(s,a)] += 1
                self.targetQsa[(s,a)] = max(self.reward + self.args['discount_factor']*v, self.targetQsa[(s,a)])


                # to get best action later on...
                temp_v = str(self.Wsa[(s,a)])
                self.Asav[(s,a,temp_v)] = self.actions_to_leafnode_nostring[i]

            else:
                self.Wsa[(s,a)] = v
                self.Nsa[(s,a)] = 1
                self.targetQsa[(s,a)] = v

                temp_v = str(self.Wsa[(s,a)])
                self.Asav[(s,a,temp_v)] = self.actions_to_leafnode_nostring[i]
            
                self.Qsa[(s,a)] = self.Wsa[(s,a)]
            self.Ns[s] += 1
            i += 1 
  

    def take_random_step(self, state1, actions_taken):
        state = copy.deepcopy(state1)
        perspective_list = self.generate_perspective(self.args['grid_shift'], state)
        s = str(state)
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


    def rollout(self, perspective_list, state1, actions_taken):
        accumulated_reward = 0 
        v = 0
        discount = self.args['discount_factor']
        state = copy.deepcopy(state1)

        counter = 0 # For rollout longer then 1
        while True:
            counter += 1
            #Check if non trivial loop
            all_zeros = not np.any(state)
            self.qubit_matrix = state
            self.eval_ground_state()
            if self.ground_state is False:
                v += -100
                break

            # om ej terminal state
            if all_zeros is False and counter <= self.args['rollout_length']:
                perspective_list = self.generate_perspective(self.args['grid_shift'], state)
                s = str(state)
                current_state = copy.deepcopy(state)

                rand_action = self.take_random_step(current_state, actions_taken)
    
                a = str(rand_action)
                self.step(rand_action, state, actions_taken)
                next_state = copy.deepcopy(state)

                
                #get reward for step
                accumulated_reward += self.get_reward(next_state, current_state)*discount**(counter)
                
                # discount factor because want to promote early high rewards
                v += accumulated_reward 
    
            #Break if terminal state
            all_zeros = not np.any(state)
            if all_zeros:
                v += discount**(counter)*100
                print('We Won in rollout! :)')
                break
            
            elif counter == self.args['rollout_length']:
                break
        return v

    def get_reward(self, next_state, current_state):
        #Check if non trivial loop
        all_zeros = not np.any(next_state)
        self.qubit_matrix = next_state
        self.eval_ground_state()
        if self.ground_state is False:
            reward = -100
        elif all_zeros:
            reward = 100
        else:
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
        if reward == 2:
            reward = 5
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