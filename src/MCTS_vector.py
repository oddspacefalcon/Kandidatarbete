import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import time

EPS = 1e-8

class MCTS_vector():

    def __init__(self, model, device, args, toric_codes=None, syndroms=None):
        self.toric_codes = toric_codes # toric_model object

        if(toric_codes == None):
            if(syndroms == None):
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndroms = syndroms
        else:
            self.syndroms = np.array([toric_code.current_state for toric_code in toric_codes])

        self.nr_trees = self.syndroms.shape[0]

        self.model = model   # resnet
        self.args = args     # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = [{} for _ in range(self.nr_trees)]        # stores Q values for s,a (as defined in the paper)
        self.Nsa = [{} for _ in range(self.nr_trees)]        # stores #times edge s,a was visited
        self.Ns = [{} for _ in range(self.nr_trees)]         # stores #times board s was visited
        self.Ps = {}       # stores initial policy (returned by neural net)
        self.device = device # 'cpu'
        self.actions = []
        self.current_level = 0
        self.loop_check = [set() for _ in range(self.nr_trees)] # set that stores state action pairs
        self.system_size = self.syndroms.shape[2]
        self.next_state = []
        self.current_state = []
        self.rewards = []

        self.Ts = np.zeros(7)
        self.Nrs = np.zeros(7)
        self.max_level = 0
        self.max_loops = 0


        '''Två arrays som används för att kolla vilka stater som ska gå igenom trädet samtidigt: '''
        self.is_terminal = np.zeros(self.nr_trees, dtype=bool)
        self.is_leaf = np.zeros(self.nr_trees, dtype = bool)
        self.v_arr = np.zeros(self.nr_trees, dtype=float)

        #self.states = copy.deepcopy(self.syndrom)


    def get_Qvals(self, temp=1):
        
        size = self.system_size
        state_string = [str(syndrom) for syndrom in self.syndroms]
        actions_taken = np.zeros((self.nr_trees,2,size,size), dtype=int)

        
        #.............................Search...............................

        for i in range(self.args['num_simulations']):

            self.search(copy.deepcopy(self.syndroms), actions_taken)
            print("sim {} is now done!".format(i))
            self.loop_check =  [set() for _ in range(self.nr_trees)]
            self.is_terminal = np.zeros(self.nr_trees, dtype=bool)
            self.is_leaf = np.zeros(self.nr_trees, dtype=bool)
        
       #..............................Max Qsa .............................
        batch_perspectives = [self.generate_perspective(self.args['grid_shift'], state) for state in self.syndroms]
        batch_perspectives = [Perspective(*zip(*perspectives)).perspective for perspectives in batch_perspectives]
        batch_actions = [self.get_possible_actions(syndrom) for syndrom in self.syndroms]
        all_Qsa = [np.array([[self.Qsa[i][(s,str(a))] if (s,str(a)) in self.Qsa[i] else 0 for a in action] for action in actions]) for actions, s, i in zip(batch_actions, state_string, range(self.nr_trees))]
        #all_Qsa = np.array([np.reshape(Qsa, Qsa.size) for Qsa in all_Qsa])
        #maxQ = [max(Qsa) for Qsa in all_Qsa]

        #Ger som output de maximala Q-värden till de olika staterna...(varför?)
        return all_Qsa, batch_perspectives, batch_actions

    def search(self, all_states, actions_taken):
        
        #...........................Check if terminal state.............................
        t0  = time.clock()
        #Check which of the states is terminal:
        #print("checking if terminal b4 and aftr: ")
        self.is_terminal[~self.is_leaf & ~self.is_terminal] = np.array([np.all(state==0) for state in all_states[~self.is_terminal & ~self.is_leaf]])
        #place the current terminal and non terminal states in seprate arrays
        #non_terminal states of those that are also not leaf states
        continuing_states = all_states[~self.is_terminal & ~self.is_leaf]

        terminal_indecies = np.where(self.is_terminal & ~self.is_leaf)[0]
        self.v_arr[self.is_terminal & ~self.is_leaf] = self.eval_ground_states(actions_taken, terminal_indecies)

        #Terminal states --> ej aktiva längre (används inte längre ned i trädet)
        

        # np.arrays unhashable, needs string

        '''Create n strings from n states'''
        state_strings = np.array([str(state) for state in continuing_states])

        # ............................If leaf node......................................
        '''av de i continuing_view --> kolla om dess stater har Q-värden här'''
        '''sedan generera alla perspektiv här...'''
        #ta bort alla de nya terminal states från leafs (om den är terminal är den ej en leafstat)
        indecies = np.where(~self.is_terminal & ~self.is_leaf)[0]
        new_is_leaf = np.array([s not in self.Ns[i] for s, i in zip(state_strings, indecies)], dtype=bool)
        #detta fungerar inte!!

        self.is_leaf[~self.is_leaf & ~self.is_terminal] = new_is_leaf

        continuing_states = continuing_states[~new_is_leaf]
        continuing_state_strings = state_strings[~new_is_leaf]

        t1 = time.clock()
        range_iter = iter(range(self.Ts.size))
        i = next(range_iter)
        self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
        self.Nrs[i]+=1

        if(len(continuing_states) == 0):
            t0 = time.clock()
            #..................Get perspectives and batch for network......................

            #From the state --> generate perspectives.
            #leaf_indecies = np.where(self.is_leaf)[0]
            leaf_states = all_states[self.is_leaf]
            leaf_state_strings = np.array([str(state) for state in leaf_states])
            leaf_mask = [s not in self.Ps for s in zip(leaf_state_strings)]



            #Processessering på stater som inte har 

            '''behöver använda states istället --> dela upp dessa [#träd, #perspectives, 2, d, d]'''
            perspective_list = [self.generate_perspective(self.args['grid_shift'], state) for state in leaf_states[leaf_mask]]

            nr_perspectives = [len(perspective) for perspective in perspective_list]

            perspectives = [Perspective(*zip(*perspective)) for perspective in perspective_list]
            batch_perspectives = [perspective.perspective for perspective in perspectives]

            #process it bettter before putting it into the GPU
            batch_perspectives = np.concatenate(batch_perspectives)
            batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
            batch_perspectives = batch_perspectives.to(self.device)
            #skapa batches från leaf_states och skikka dessa till device...
            #utför att skikka batches osv...
            '''Göra för alla möjliga olika states istället'''
            #print(batch_perspectives)
            t1 = time.clock()
            
            i = next(range_iter)
            self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
            self.Nrs[i]+=1

            t0 = time.clock()

            with torch.no_grad():
                Qs = self.model.forward(batch_perspectives)

            t1 = time.clock()
            
            i = next(range_iter)
            self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
            self.Nrs[i]+=1

            t0 = time.clock()

            
            
            #Delar upp Qvärdena för de olika perspektiven i de olika träden de tillhör
            perspective_indecies = [(sum(nr_perspectives[0:i]), sum(nr_perspectives[0:i+1])) for i in range(len(nr_perspectives))]
             
            Qs_per_tree = [Qs.detach().numpy()[i:j] for i, j in perspective_indecies]


            Ps_array = [Q/np.sum(Q) if mask else self.Ps[s] for Q, mask, s in zip(Qs_per_tree, leaf_mask, leaf_state_strings)]

            indecies = np.where(self.is_leaf)[0]

            for Ps, s, i in zip(Ps_array, leaf_state_strings, indecies):
                self.Ps[s] = Ps
                self.Ns[i][s] = 0
            self.v_arr[self.is_leaf] = [np.max(Q) for Q in Qs_per_tree]

            t1 = time.clock()

            i = next(range_iter)
            self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
            self.Nrs[i]+=1

            return
        next(range_iter)
        next(range_iter)
        next(range_iter)

        # ..........................Get best action...................................
        t0 = time.clock()
        #Göra använda numpy för att göra UBC kalkyleringen snabbare.
        '''Behöver kolla igenom alla icke leaf nodes'''
        '''Behöver sedan stoppa inn en array med Ps[s]'s och array med states (icke-leaf-nodes)'''
        perspective_list = [self.generate_perspective(self.args['grid_shift'], state) for state in continuing_states]
        number_of_perspectives = [len(perspective) for perspective in perspective_list]
        perspectives = [Perspective(*zip(*perspective)) for perspective in perspective_list]
        batch_perspectives = [np.array(perspective.perspective) for perspective in perspectives]

        batch_position_actions = [perspective.position for perspective in perspectives]

        actions = [[[Action(np.array(p_pos), x) for x in range(1,4)] for p_pos in position_actions] for position_actions in batch_position_actions]

        Ps_arr = np.array([self.Ps[s] for s in continuing_state_strings])


        indecies = np.where(~self.is_leaf & ~self.is_terminal)[0]

        UpperConfidence = self.UCBpuct(Ps_arr, actions, continuing_state_strings, indecies)

        #Väljer ut action med högst UCB som inte har valts tidigare (i denna staten):

        t1 = time.clock()
            
        i = next(range_iter)
        self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
        self.Nrs[i]+=1

        t0 = time.clock()

        UC = UpperConfidence
        is_complete = np.zeros(len(UC), dtype=bool)
        best_perspectives = np.array([None for _ in range(len(UC))])
        best_actions = np.array([None for _ in range(len(UC))])
        inf_UBC_mask = np.zeros(len(UC), dtype=bool)

        '''Tar bort actions som har gjorts tidigare'''
        nr_loops = 0
        while len(UC) != 0:
            nr_loops+=1
            perspective_indecies, action_indecies = zip(*[np.unravel_index(np.argmax(U), U.shape) for U in UC])
            perspective_indecies = np.array(perspective_indecies)
            action_indecies = np.array(action_indecies)


            non_complete_indecies = np.where(~is_complete)[0]
            best_perspectives[non_complete_indecies] = [perspective_list[i][index] for index, i in zip(perspective_indecies, non_complete_indecies)]
            best_actions[non_complete_indecies] = [Action(np.array(best_perspective.position), action_index+1) for best_perspective, action_index in zip(best_perspectives, action_indecies)]
            
            
            #string_states = [str(perspective.perspective) for perspective in best_perspectives[~is_complete]]

            inf_UBC_mask[~is_complete] = np.array([UC[i][pi][ai] == -float('inf') for i, pi, ai in zip(range(len(action_indecies)), perspective_indecies, action_indecies)])

            string_actions = [str(action) for action in best_actions[~is_complete]]
            new_is_complete = [a not in self.loop_check[i] for a, i in zip(string_actions, range(len(string_actions)))] | inf_UBC_mask[~is_complete]
            is_complete[~is_complete] = new_is_complete

            '''Först: ger lista av alla element som har högst UBC för respektive träd:
                Sedan: plockar ut element som är i loop check  (element (s,a) som redan har besökts) --> sätter dessa till -float('inf')
            '''
            non_complete_indecies = np.where(~is_complete)[0]


            #UpperConfidence till de perspektiven som man nu innser redan har varit besökta sätts till -float('inf')
            '''Bör evt förändras...'''
            for i, pi, ai in zip(non_complete_indecies, perspective_indecies[~new_is_complete], action_indecies[~new_is_complete]):
                UpperConfidence[i][pi][ai] = -float('inf')
            UC = UpperConfidence[~is_complete]

            complete_indecies = np.where(is_complete)[0]
            string_actions = [str(action) for action in best_actions[is_complete]]
            for i, a in zip(complete_indecies, string_actions):
                self.loop_check[i].add(a)
        '''Borde eventuellt ta bort denna (skapa hela i loopen ovan):'''
        if(self.max_loops < nr_loops):
            self.max_loops = nr_loops

        self.is_terminal[indecies] = inf_UBC_mask
        self.v_arr[indecies[inf_UBC_mask]] = -1
        continuing_states = np.delete(continuing_states, np.where(inf_UBC_mask)[0], axis=0)
        continuing_state_strings = np.delete(continuing_state_strings, np.where(inf_UBC_mask)[0], axis=0)
        best_actions = np.delete(best_actions, np.where(inf_UBC_mask)[0], axis=0)

        t1 = time.clock()
            
        i = next(range_iter)
        self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
        self.Nrs[i]+=1

        #Om alla stater försvinner pga de har användt alla moves -> returnera
        if(len(continuing_states) == 0):
            return

        string_actions = [str(action) for action in best_actions]
        
        #går ett steg framåt

        for i, action in zip(indecies, best_actions):
            self.step(action, all_states[i], actions_taken[i])
        self.current_level += 1
        if self.current_level == 1:
            for a in string_actions:
                self.actions.append(a)
        
        if(self.current_level > self.max_level):
            self.max_level = self.current_level
        
        #kollar igenom nästa state
        
        self.search(all_states, actions_taken)
        t0 = time.clock()
        #går tilbaka med samma steg och finn rewards
        
        '''Fattar inte varför man använder self.reward'''
        self.next_states = []
        self.current_states = []
        self.rewards = []
        for state in all_states[~self.is_terminal & ~self.is_leaf]:
            self.next_states.append(copy.deepcopy(state))
        for state, action, action_taken in zip(all_states[~self.is_terminal & ~self.is_leaf], best_actions, actions_taken):
            state = self.step(action, state, action_taken)
            self.current_states.append(copy.deepcopy(state))
            self.rewards.append(0.3*self.get_reward(self.next_state, self.current_state))

        self.current_level -= 1
        
        
        
        #Obs 0.3*reward pga annars blir denna för dominant -> om negativ -> ibland negativt Qsa
        
        
        

        # ............................BACKPROPAGATION................................
        
        for s, a, i, reward in zip(continuing_state_strings, string_actions, indecies, self.rewards):
            if (s,a) in self.Qsa[i]:
                
                self.Qsa[i][(s,a)] = (self.Nsa[i][(s,a)]*self.Qsa[i][(s,a)] + reward + self.args['disscount_factor']*self.v_arr[i])/(self.Nsa[i][(s,a)]+1)
                #print('Qsa: ',self.Qsa[(s,a)])
                #print('reward:', self.reward)
                self.Nsa[i][(s,a)] += 1
            else:
                self.Qsa[i][(s,a)] = self.v_arr[i]
                self.Nsa[i][(s,a)] = 1
                self.Ns[i][s] += 1
        t1 = time.clock()
            
        i = next(range_iter)
        self.Ts[i] = (self.Ts[i]*self.Nrs[i]+t1-t0)/(self.Nrs[i]+1)
        self.Nrs[i]+=1
        return


    def next_step(self, actions):
        for action, i in zip(actions, range(len(actions))):
            self.toric_codes[i].step(action)
            self.syndroms[i] = self.toric_codes[i].next_state

    def get_best_indices(self, Qvals):
        perspective_indices, action_indices = zip(*[np.unravel_index(np.argmax(Qs), Qs.shape) for Qs in Qvals])
        perspective_indices = np.array(perspective_indices)
        action_indices = np.array(action_indices)
        return (perspective_indices, action_indices)



    # Reward
    '''Får skriva om denna så att den tar alla samtidgt '''
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

    def UCBpuct(self, probability_matrix, batch_actions, batch_s, indecies):

        current_Qsa = np.array([np.array([[self.Qsa[(s,str(a))] if (s, str(a)) in self.Qsa else 0 for a in opperator_actions] for opperator_actions in actions]) for s, actions in zip(batch_s, batch_actions)])
        current_Nsa = np.array([np.array([[self.Nsa[(s,str(a))] if (s, str(a)) in self.Nsa else 0 for a in opperator_actions] for opperator_actions in actions]) for s, actions in zip(batch_s, batch_actions)])

        current_Ns = np.array([0.001 if s not in self.Ns[i] else (self.Ns[i][s] if self.Ns[i][s] != 0 else 0.001) for i, s in zip(indecies, batch_s)])
        
        UBC = []
        for Ns, Nsa, Qsa, Ps in zip(current_Ns, current_Nsa, current_Qsa, probability_matrix):
            UBC.append(Qsa + self.args['cpuct']*Ps*np.sqrt(Ns/(1+Nsa)))
        return np.array(UBC)


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
        return syndrom
    
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

        return np.array([[[rule_table[qu1][qu2] for qu1, qu2 in zip(row1, row2)] for row1, row2 in zip(qu_mat1, qu_mat2)] for qu_mat1, qu_mat2 in zip(action_matrix1, action_matrix2)])
    
    def get_possible_actions(self, state):
        perspectives = self.generate_perspective(self.args['grid_shift'], state)
        perspectives = Perspective(*zip(*perspectives))
        return [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position] #bytte ut np.array(p_pos)
    
    def eval_ground_states(self, actions_taken, indecies):
        if(self.toric_codes!=None):
            ans_list = []
            toric_list = np.array(self.toric_codes)[indecies]
            for at, toric in zip(actions_taken, toric_list):
                toric.qubit_matrix = self.mult_actions(toric.qubit_matrix, at)
                assert toric.qubit_matrix.shape == (2, self.system_size, self.system_size)
                ans_list.append(1 if toric.eval_ground_state() else -1)
                toric.qubit_matrix = self.mult_actions(toric.qubit_matrix, at)
            return ans_list
        else:
            return [1 for _ in range(len(indecies))]
        