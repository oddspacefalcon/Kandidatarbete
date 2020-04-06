import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
EPS = 1e-8
import time

class MCTS():

    def __init__(self, model, device, args, toric_code=None, syndrom=None):
        self.toric_code = toric_code # toric_model object

        if(toric_code == None):
            if(syndrom is None):
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
        else:
            self.syndrom = self.toric_code.current_state

        self.model = model   # resnet
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

        self.Ts = np.zeros(5)
        self.Nrs = np.zeros(5)


    def get_Qvals(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)

        #.............................Search...............................

        for i in range(self.args['num_simulations']):
            self.search(copy.deepcopy(self.syndrom), actions_taken)
            self.loop_check.clear()
        
       #.............................. Return Q_vals and corresponding actions and perspectives .............................
        actions = self.get_possible_actions(self.syndrom)
        perspectives = self.generate_perspective(self.args['grid_shift'], self.syndrom)
        perspectives = Perspective(*zip(*perspectives)).perspective

        all_Qsa = np.array([[self.target_Qsa[(s,str(a))] if (s,str(a)) in self.target_Qsa else 0 for a in position] for position in actions])
        
        return (all_Qsa, perspectives, actions)

    #Takes in all the Qvalues and the corresponding perspectives and actions
    # and returns the ones that have been visited more than or equal to nr_min_visits
    def get_memory_Qvals(self, Qvals, perspectives, actions, nr_min_visits=5):
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
            

    def search(self, state, actions_taken):
        # np.arrays unhashable, needs string
        t0 = time.clock()
        s = str(state)

        #..................Get perspectives and batch for network......................

        perspective_list = self.generate_perspective(self.args['grid_shift'], state)
        number_of_perspectives = len(perspective_list)
        perspectives = Perspective(*zip(*perspective_list))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        
        #...........................Check if terminal state.............................

        if np.all(state == 0):
            if state.eval_ground_state(): #ska det vara self.toric.eval_ground_state(): här istället?
                #Trivial loop --> gamestate won!
                return 1
            else:
                #non trivial loop --> game lost!
                return -1

        # ............................If leaf node......................................
        self.Ts[0] = (self.Ts[0]*self.Nrs[0]+time.clock()-t0)/(self.Nrs[0]+1)
        self.Nrs[0] += 1

        t0 = time.clock()
        if s not in self.Ps:
            # leaf node => expand
            with torch.no_grad():
                self.Ps[s] = self.model.forward(batch_perspectives) 
            v = torch.max(self.Ps[s])
            v = v.data.numpy()
            # Normalisera
            sum_Ps_s = torch.sum(self.Ps[s]) 
            self.Ps[s] = self.Ps[s]/sum_Ps_s    
                
            # ej besökt detta state tidigare sätter dessa parametrar till 0
            self.Ns[s] = 0
            return v

        self.Ts[1] = (self.Ts[1]*self.Nrs[1]+time.clock()-t0)/(self.Nrs[1]+1)
        self.Nrs[1] += 1
        # ..........................Get best action...................................
        t0 = time.clock()
        #Göra använda numpy för att göra UBC kalkyleringen snabbare.
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
        UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)

        self.Ts[2] = (self.Ts[2]*self.Nrs[2]+time.clock()-t0)/(self.Nrs[2]+1)
        self.Nrs[2] += 1
        #Väljer ut action med högst UCB som inte har valts tidigare (i denna staten):
        t0 = time.clock()
        while True:
            t0 = time.clock()
            maxindex = np.argmax(UpperConfidence)
            self.Ts[3] = (self.Ts[3]*self.Nrs[3]+time.clock()-t0)/(self.Nrs[3]+1)
            self.Nrs[3] += 1
            perspective_index, action_index = np.unravel_index(maxindex, UpperConfidence.shape)
            
            best_perspective = perspective_list[perspective_index]

            action = Action(np.array(best_perspective.position), action_index+1) #bytte ut np.array(best_perspective.position)
            

            a = str(action)
            if(a not in self.loop_check):
                self.loop_check.add(a)
                break
            else:
                UpperConfidence[perspective_index][action_index] = -float('inf')
    
        #går ett steg framåt
        
        self.step(action, state, actions_taken)
        self.current_level += 1
        if self.current_level == 1:
            self.actions.append(a)
            
        
        #kollar igenom nästa state
        
        v = self.search(state, actions_taken)
        
        #går tilbaka med samma steg och finn rewards
        self.next_state = copy.deepcopy(state)
        
        self.step(action, state, actions_taken)
        self.current_level -= 1
        self.current_state = copy.deepcopy(state)
        #Obs 0.3*reward pga annars blir denna för dominant -> om negativ -> ibland negativt Qsa
        self.reward = 0.3*self.get_reward(self.next_state,self.current_state, actions_taken)
        
        
        # ............................BACKPROPAGATION................................

        if (s,a) in self.Qsa:
            
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + self.reward + self.args['discount_factor']*v)/(self.Nsa[(s,a)]+1)
            new_Qval = self.reward/0.3 + self.args['discount_factor']*v
            self.target_Qsa[(s,a)] = new_Qval if new_Qval > self.target_Qsa[(s,a)] else self.target_Qsa[(s,a)]
            #print('Qsa: ',self.Qsa[(s,a)])
            #print('reward:', self.reward)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.target_Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1

        
        return v

    # Reward
    def get_reward(self, next_state, current_state, action_matrix):
        
        terminal = np.count_nonzero(next_state)==0
        
        if terminal == True:
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
            
            t0 = time.clock()
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
            self.Ts[4] = (self.Ts[4]*self.Nrs[4]+time.clock()-t0)/(self.Nrs[4]+1)
            self.Nrs[4] += 1
            #print('Reward = ', reward)
        return reward

    def best_index(self, Qvals):
        return np.unravel_index(np.argmax(Qvals), Qvals.shape)

    def next_step(self, action):
        self.toric_code.step(action)
        self.syndrom = self.toric_code.next_state

    def UCBpuct(self, probability_matrix, actions, s):

        current_Qsa = np.array([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions])
        current_Nsa = np.array([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions])
        if s not in self.Ns:
            current_Ns = 0.001
        else:
            if self.Ns[s] == 0:
                current_Ns = 0.001
            else:
                current_Ns = self.Ns[s]
        #använd max Q-värde: (eller använda )
        return current_Qsa + self.args['cpuct']*probability_matrix.detach().numpy()*np.sqrt(current_Ns/(1+current_Nsa))


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
