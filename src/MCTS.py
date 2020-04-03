import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import time

EPS = 1e-8

class MCTS():

    def __init__(self, model, device, args, toric_code=None, syndrom=None):

        self.toric_code = toric_code # toric_model object

        if toric_code is None:
            if syndrom is None:
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
        else:
            self.syndrom = self.toric_code.current_state


        self.model = model   # resnet
        self.args = args     # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = {}        # stores Q values for s,a (as defined in the paper)
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
        self.atLeafNode = False

    
    def get_qs_actions(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)

        #.............................Search...............................

        for i in range(self.args['num_simulations']):
            
            self.search(copy.deepcopy(self.syndrom), actions_taken)
            self.loop_check.clear()
            
        
       #..............................Max Qsa .............................
        actions = self.get_possible_actions(self.syndrom)
        all_Qsa2D = np.array([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])
        all_Qsa = np.reshape(all_Qsa2D, all_Qsa2D.size)
        maxQ = max(all_Qsa != 0 )
        
        #best action
        index_max = np.unravel_index((all_Qsa2D).argmax(), all_Qsa2D.shape)
        best_action = actions[index_max[0]][index_max[1]]
        print(max(self.Qsa, key=self.Qsa.get))

        return maxQ, all_Qsa, best_action

    def search(self, state, actions_taken):
        with torch.no_grad():
            # np.arrays unhashable, needs string
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
    
            if s not in self.Ps:
                # leaf node => expand
                #start = time.time()
                self.Ps[s] = self.model.forward(batch_perspectives.cuda()) 
                #end = time.time()
                #print(end-start)
                #print('-------')
                v = torch.max(self.Ps[s]).cpu()
                v = v.data.numpy()

                # Normalisera
                sum_Ps_s = torch.sum(self.Ps[s]) 
                self.Ps[s] = self.Ps[s]/sum_Ps_s    
                    
                # ej besökt detta state tidigare sätter dessa parametrar till 0
                self.Ns[s] = 0
                return v
    
            # ..........................Get best action...................................
    
            #Göra använda numpy för att göra UBC kalkyleringen snabbare.
            actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
            UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)
    
            #Väljer ut action med högst UCB som inte har valts tidigare (i denna state):
            while True:
                perspective_index, action_index = np.unravel_index(np.argmax(UpperConfidence), UpperConfidence.shape)
                best_perspective = perspective_list[perspective_index]
    
                action = Action(np.array(best_perspective.position), action_index+1) #bytte ut np.array(best_perspective.position)
    
                a = str(action)
                if((s,a) not in self.loop_check):
                    self.loop_check.add((s,a))
                    break
                else:
                    UpperConfidence[perspective_index][action_index] = -float('inf')
        
            
            #går ett steg framåt
            self.step(action, state, actions_taken)
            self.current_level += 1
            if self.current_level == 1:
                self.actions.append(a)
                
            
            #Får v för leaf node
            v = self.search(state, actions_taken)

            
            #går tilbaka med samma steg och finn rewards
            self.next_state = copy.deepcopy(state)
            self.step(action, state, actions_taken)
            self.current_level -= 1
            self.current_state = copy.deepcopy(state)
            #Obs om reward<0 sätt lika med 0  eftersom om negativ reward -> ibland negativt Qsa
            self.reward = self.get_reward(self.next_state,self.current_state)
            if self.reward < 0:
                self.reward=0
    
            #print(self.reward)
            # ............................BACKPROPAGATION................................
    
            if (s,a) in self.Qsa:
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + self.reward + self.args['disscount_factor']*v)/(self.Nsa[(s,a)]+1)
                #print('Qsa: ',self.Qsa[(s,a)])
                self.Nsa[(s,a)] += 1
            else:
                self.Qsa[(s,a)] = v
                self.Nsa[(s,a)] = 1
    
            self.Ns[s] += 1
        return v

    # Reward
    def get_reward(self, next_state, current_state):
        terminal = np.all(next_state==0)
        if terminal == True:
            reward = 100
            #print('Reward = ', reward)
 
        else:
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
            #print('Reward = ', reward)
        return reward

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
        
        return current_Qsa + self.args['cpuct']*probability_matrix.cpu().data.numpy()*np.sqrt(current_Ns/(1+current_Nsa))


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