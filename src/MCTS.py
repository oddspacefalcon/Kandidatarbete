import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
EPS = 1e-8

class MCTS():
    """
    Represents a MCTS tree
    """

    def __init__(self, model, device, args, toric_code=None, syndrom=None):
        self.toric_code = toric_code # toric_model object

        if(toric_code == None):
            if(syndrom == None):
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
        else:
            self.syndrom = self.toric_code.current_state

        self.model = model  # resnet
        self.args = args    # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.device = device # 'cpu'
        self.actions = []
        self.current_level = 0
        self.loop_check = set() # set that stores state action pairs
        self.system_size = self.syndrom.shape[1]

    def search(self, state, actions_taken):
        # np.arrays unhashable, needs string
        s = str(state)

        perspective_list = self.generate_perspective(self.args['grid_shift'], state)
        number_of_perspectives = len(perspective_list)
        perspectives = Perspective(*zip(*perspective_list))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        if np.all(state == 0):
            # if terminal state
            return 1 # terminal <=> vunnit
            '''
            #Om vi vill använda rewards på slutet (100 för trivial -100 för icke trivial)
            # Görs ändast om vi har lagt till toric model som input 

            if(self.toric_code == None):
                return 1
            else:
                qubit_matrix = self.toric_code.qubit_matrix
                self.toric_code.qubit_matrix = self.mult_actions(qubit_matrix, actions_taken)
                r = 100 if self.toric_code.eval_ground_state() else -100
                self.toric_code.qubit_matrix = qubit_matrix
                return r
            '''
            #--> Borde returnera negativ om man gör fel...

        if s not in self.Ps:
            # leaf node => expand
            self.Ps[s], v = self.model.forward(batch_perspectives)
            self.Ns[s] = 0
            return v


        #Göra använda numpy för att göra UBC kalkyleringen snabbare.
        actions = [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
        UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)

        
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

        #kollar igenom nästa stat
        v = self.search(state, actions_taken)

        #går tilbaka med samma steg
        self.step(action, state, actions_taken)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v

    def get_probs_action(self, temp=1):

        size = self.system_size
        s = str(self.syndrom)
        actions_taken = np.zeros((2,size,size), dtype=int)

        for i in range(self.args['num_simulations']):
             # if not copied the operations on the toric code would be saved over every tree
             #instead add toric_codes syndrom
            self.search(copy.deepcopy(self.syndrom), actions_taken)
             # clear loop_check so the same path can be taken in new tree
            self.loop_check.clear()

        #kollar inte alla möjliga actions
        actions = self.get_possible_actions(self.syndrom)

        counts = np.array([[self.Nsa[(s,str(a))] if (s,str(a)) in self.Nsa else 0 for a in position] for position in actions])
        counts = np.reshape(counts, counts.size)

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs, actions[bestA]

        #evt välja största Q-värde
        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        probs = torch.tensor(probs)
        # sample an action according to probabilities probs
        action = torch.multinomial(probs, 1)
        probs = probs.view(-1, 3)
        return probs, actions[action]

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

    def UCBpuct(self, probability_matrix, actions, s):
        #s = qubit-matrisen
        #en annan fråga är ochså hur jag representerar actions här. Vanligtvis brukar detta göras med att man räknar ut alla
        #Borde hitta nogont set datatyp som låter en göra parallela acions på flera state action pairs samtidigt
        #Borde också eventuellt hitta ett bättre sätt att representera actions på ett bättre sätt
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
        return [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]
