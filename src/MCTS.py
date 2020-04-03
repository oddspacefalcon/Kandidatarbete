import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
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
        self.device = device
        self.loop_check = set() # set that stores state action pairs
        self.system_size = self.syndrom.shape[1]
        self.next_state = []
        self.current_state = []
        self.batch_size = self.syndrom.shape[0]
        self.targetQsa = {}

        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.ground_state = True    # True: only trivial loops, 
                                   # False: non trivial loop

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
        all_Qsa = torch.tensor([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])

        return all_Qsa, actions, self.targetQsa

    def search(self, state, actions_taken):
        with torch.no_grad():
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
                self.Ps[s] = self.model(batch_perspectives) 
                v = torch.max(self.Ps[s])

                # Normalisera. # Fungerar detta iom att nätverket outputar negativa tal också? Relu på output?
                sum_Ps_s = torch.sum(self.Ps[s]) 
                self.Ps[s] = self.Ps[s]/sum_Ps_s    

                # ej besökt detta state tidigare sätter dessa parametrar till 0
                self.Ns[s] = 0
                return v

            # ..........................Get best action...................................

            #Göra använda numpy för att göra UBC kalkyleringen snabbare.
            actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
            UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)

            #Väljer ut action med högst UCB som inte har valts tidigare (i denna staten):
            while True:
                # omvandlar 1D index -> 2D index
                argmax = torch.argmax(UpperConfidence).cpu().numpy()
                perspective_index = argmax // UpperConfidence.shape[1]
                action_index = argmax % UpperConfidence.shape[1]
                best_perspective = perspective_list[perspective_index]

                action = Action(np.array(best_perspective.position), action_index+1)

                a = str(action)

                if (s,a) not in self.loop_check:
                    self.loop_check.add((s,a))
                    break
                else:
                    UpperConfidence[perspective_index][action_index] = -float('inf')
        
            #går ett steg framåt
            self.step(action, state, actions_taken)                
            
            #kollar igenom nästa state
            v = self.search(state, actions_taken)

            #går tilbaka med samma steg och finn rewards
            self.next_state = copy.deepcopy(state)
            self.step(action, state, actions_taken)
            self.current_state = copy.deepcopy(state)
            
            #Obs 0.3*reward pga annars blir denna för dominant -> om negativ -> ibland negativt Qsa
            self.reward = self.get_reward(self.next_state,self.current_state)
            if self.reward < 0:
                self.reward = self.reward*0.3
            if self.reward == 0:
                self.reward = -EPS
            
            # ............................BACKPROPAGATION................................

            if (s,a) in self.Qsa:
                self.targetQsa[(s,a)] = max(self.reward + self.args['discount_factor']*v, self.targetQsa[(s,a)])
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + self.reward + self.args['discount_factor']*v)/(self.Nsa[(s,a)]+1)                
                self.Nsa[(s,a)] += 1
            else:
                self.targetQsa[(s,a)] = v
                self.Qsa[(s,a)] = v
                self.Nsa[(s,a)] = 1

            self.Ns[s] += 1
        return v 

    def get_reward(self, next_state, current_state):
        #Check if non trivial loop
        all_zeros = not np.any(state)
        self.qubit_matrix = state
        self.eval_ground_state()
        if self.ground_state is False:
            reward = -100
        elif all_zeros == True:
            reward = 100
        else:
            defects_state = np.sum(current_state)
            defects_next_state = np.sum(next_state)
            reward = defects_state - defects_next_state
        return reward

    def UCBpuct(self, probability_matrix, actions, s):
        current_Qsa = torch.tensor([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions], dtype=torch.float32, device=self.device)
        current_Nsa = torch.tensor([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions], dtype=torch.float32, device=self.device)
        
        if s not in self.Ns:
            current_Ns = 0.001
        else:
            if self.Ns[s] == 0:
                current_Ns = 0.001
            else:
                current_Ns = self.Ns[s]
        
        #använd max Q-värde: (eller använda )
                
        return current_Qsa + self.args['cpuct']*probability_matrix*torch.sqrt(torch.tensor(current_Ns, dtype=torch.float32)/(1+current_Nsa))

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