import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
EPS = 1e-8
import time

class MCTS_rollout3():

    def __init__(self, device, args, toric_code=None, syndrom=None):
        self.toric_code = toric_code # toric_model object

        if(toric_code == None):
            if(syndrom is None):
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
        else:
            self.syndrom = self.toric_code.current_state

        self.args = args     # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = {}        # stores Q values for s,a (as defined in the paper)
        self.target_Qsa = {}
        self.Nsa = {}        # stores #times edge s,a was visited
        self.Ns = {}         # stores #times board s was visited
        self.Ps = {}         # stores initial policy (returned by neural net)
        self.device = device # 'cpu'
        self.actions = []
        self.current_level = 0
        self.loop_check = set() # set that stores state action pairs
        self.system_size = self.syndrom.shape[1]
        self.next_state = []
        self.current_state = []
        self.myS = ""

    def get_Qvals(self, temp=1):

        size = self.system_size
        s = self.syndrom.tostring()
        self.myS = s
        actions_taken = np.zeros((2,size,size), dtype=int)

        #.............................Search...............................
        for i in range(self.args['num_simulations']):
            print(i)
            self.search(copy.deepcopy(self.syndrom), actions_taken)
            self.loop_check.clear()
       #.............................. Return Q_vals and corresponding actions and perspectives .............................
        actions = self.get_possible_actions(self.syndrom)
        perspectives = self.generate_perspective(self.args['grid_shift'], self.syndrom)
        perspectives = Perspective(*zip(*perspectives)).perspective
        all_Qsa = self.target_Qsa[s]

        return (all_Qsa, perspectives, actions)

    #Takes in all the Qvalues and the corresponding perspectives and actions
    # and returns the ones that have been visited more than or equal to nr_min_visits
    def get_memory_Qvals(self, Qvals, perspectives, actions, nr_min_visits=5):
        new_Qval = []
        new_perspectives = []

        s = self.syndrom.tostring()

        for i, list_action in zip(range(len(Qvals)), actions):
            nr_visit_sum = np.sum(self.Nsa[s][i])
            if nr_visit_sum >= nr_min_visits:
                new_Qval.append(Qvals[i])
                new_perspectives.append(perspectives[i])
        return (new_Qval, new_perspectives)
            

    #MCTS:
    def search(self, state, actions_taken):
        # np.arrays unhashable, needs string
        s = state.tostring()

        #..................Get perspectives and batch for network......................
        perspective_pos_list = self.generate_perspective_pos(state)
        number_of_perspectives = perspective_pos_list.shape[0]
        perspective_list = self.generate_perspective(self.args['grid_shift'], state)
        perspectives = Perspective(*zip(*perspective_list))
        
        #...........................Check if terminal state.............................
        if np.all(state == 0):
            if self.eval_ground_state(state): #ska det vara self.toric.eval_ground_state(): här istället?
                #Trivial loop --> gamestate won!
                return 100
            else:
                #non trivial loop --> game lost!
                return -100

        # ............................If leaf node......................................
        if s not in self.Ns:
            # leaf node => expand
            v = self.rollout(perspective_list, copy.deepcopy(state), actions_taken)          
              
            # ej besökt detta state tidigare sätter dessa parametrar till 0
            self.Ns[s] = 0
            return v

        # ..........................Get best action...................................
        UpperConfidence = self.UCBpuct(perspectives, s)
        #Väljer ut action med högst UCB som inte har valts tidigare (i denna staten):
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
        #går ett steg framåt
        self.step(action, state, actions_taken)
        self.current_level += 1
        
        #kollar igenom nästa state
        v = self.search(state, actions_taken)
        
        #går tilbaka med samma steg och finn rewards
        self.next_state = copy.deepcopy(state)
        self.step(action, state, actions_taken)
        
        self.current_level -= 1
        self.current_state = state
        
        #Obs 0.3*reward pga annars blir denna för dominant -> om negativ -> ibland negativt Qsa
        self.reward = 0.3*self.get_reward(self.next_state,self.current_state, actions_taken)
        
        # ............................BACKPROPAGATION................................

        ai = action_index
        pi = perspective_index
        #if(self.current_level==0 and self.myS != s):
            #print("not same S!\nmyS: {0}\ns: {1}".format(self.myS, s))
        #else:
            ##print("same S")

        if s in self.Qsa:
            self.Qsa[s][pi][ai] = (self.Nsa[s][pi][ai]*self.Qsa[s][pi][ai] + self.reward + self.args['discount_factor']*v)/(self.Nsa[s][pi][ai]+1)
            new_Qval = self.reward/0.3 + self.args['discount_factor']*v
            self.target_Qsa[s][pi][ai] = new_Qval if new_Qval > self.target_Qsa[s][pi][ai] else self.target_Qsa[s][pi][ai]
            #print('Qsa: ',self.Qsa[(s,a)])
            #print('reward:', self.reward)
            self.Nsa[s][pi][ai] += 1
        else:
            self.Qsa[s] = np.zeros((number_of_perspectives, 3))
            self.Qsa[s][pi][ai] = v
            self.target_Qsa[s] = np.zeros((number_of_perspectives, 3))
            self.target_Qsa[s][pi][ai] = v
            self.Nsa[s] = np.zeros((number_of_perspectives, 3))
            self.Nsa[s][pi][ai] = 1

        self.Ns[s] += 1

        return v


    def take_random_step(self, state, actions_taken):
        perspective_list = self.generate_perspective(self.args['grid_shift'], state)
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


    def rollout(self, perspective_list, state, actions_taken):
        accumulated_reward = 0 
        v = 0
        discount = self.args['discount_factor']
        counter = 0 # For rollout longer then 1
        while True:
            counter += 1

            #if no trivial loop check if terminal state
            defects_next_state = np.count_nonzero(state)
            if defects_next_state==0:
                if self.toric_code != None:
                    if self.toric_code.eval_ground_state():
                        v += 100
                        break
                    else: 
                        v += -100
                        break
                else:
                    v += 100
                    break
    
            # om ej terminal state
            if defects_next_state != 1 and counter <= self.args['rollout_length']:
                perspective_list = self.generate_perspective(self.args['grid_shift'], state)
                s = str(state)
                rand_action = self.take_random_step(state, actions_taken)
                a = str(rand_action)
  
                current_state = copy.deepcopy(state)
                self.step(rand_action, state, actions_taken)
                next_state = copy.deepcopy(state)

                #get reward for step
                accumulated_reward += self.get_reward(next_state, current_state, actions_taken)*discount**(counter)
                
                # discount factor because want to promote early high rewards
                v += accumulated_reward 
    
            #Break if terminal state
            #if no trivial loop check if terminal state
            defects_next_state = np.count_nonzero(state)
            if defects_next_state==0:
                if self.toric_code != None:
                    if self.toric_code.eval_ground_state():
                        v += 100
                        break
                    else: 
                        v += -100
                        break
                else:
                    v += 100
                    break

            if counter == self.args['rollout_length']:
                break
        return v

    # Reward
    def get_reward(self, next_state, current_state, action_matrix):
        defects_next_state = np.count_nonzero(next_state)
        if defects_next_state==0:
            if self.toric_code != None:
                state = self.mult_actions(self.toric_code.qubit_matrix, action_matrix)
                if self.toric_code.eval_ground_state():
                    reward = 100
                else: 
                    reward = -100
            else:
                reward = 100
            #print('Reward = ', reward)
            
 
        else:
            defects_state = np.count_nonzero(current_state)
            reward = defects_state - defects_next_state
            #print('Reward = ', reward)
        
        return reward

    def best_index(self, Qvals):
        return np.unravel_index(np.argmax(Qvals), Qvals.shape)

    def next_step(self, action):
        self.toric_code.step(action)
        self.syndrom = self.toric_code.next_state

    def UCBpuct(self, perspectives, s):
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
        current_Qsa = np.array([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions])
        current_Nsa = np.array([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions])
        if s not in self.Ns:
            current_Ns = 1.0001
        else:
            if self.Ns[s] == 0:
                current_Ns = 1.0001
            else:
                current_Ns = self.Ns[s]
        #använd max Q-värde: (eller använda )
        return current_Qsa + self.args['cpuct']*np.sqrt(np.log(current_Ns)/(current_Nsa+EPS))


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
    
    def generate_perspective_pos(self, state):
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        qubit_mask = np.zeros(state.shape, dtype=bool)
        qubit_mask[0,:,:] = (plaquette_matrix == 1) | (vertex_matrix == 1) | (np.roll(vertex_matrix, -1, axis=0) == 1) | (np.roll(plaquette_matrix, 1, axis=1) == 1)
        qubit_mask[1,:,:] = (plaquette_matrix == 1) | (vertex_matrix == 1) | (np.roll(vertex_matrix, -1, axis=1) == 1) | (np.roll(plaquette_matrix, 1, axis=0) == 1)
        perspective_pos = np.where(qubit_mask)
        return np.array([*zip(*perspective_pos)])
    

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
                syndrom[0][row][col] = (syndrom[1][row][col]+1)%2
                syndrom[0][(row-1)%self.system_size][col] = (syndrom[1][(row-1)%self.system_size][col]+1)%2
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

    def eval_ground_state(self, state):
        return True