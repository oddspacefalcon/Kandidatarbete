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
        self.device = device # 'cpu' or 'cuda'
        self.actions = []
        self.loop_check = set() # set that stores state action pairs
        self.system_size = self.syndrom.shape[1]

        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.ground_state = True    # True: only trivial loops, 
                                   # False: non trivial loop 

        self.states_to_leafnode = [] # states from root to leaf
        self.actions_to_leafnode = [] #Actions from root to leaf
        self.actions_to_leafnode_nostring = []
        self.reward = 0 #save reward when we go to leaf node
        
        ####
        self.last_best_action_array = [] # best action from previous MCTS run
        ####

        self.num_backprop = 0
        self.layer = 0
    

    def get_qs_actions(self):

        size = self.system_size
        s = str(self.toric_code.current_state)
        actions_taken = np.zeros((2,size,size), dtype=int)
        
        #.............................Search...............................
        
        for i in range(self.args['num_simulations']):
            #print('__________________')
            #print('__________________')
            #print('Root layer:', self.layer, ' Sim nr:', i)
            #print('Sim nr:', i)
            self.search(copy.deepcopy(self.syndrom), actions_taken, str(copy.deepcopy(self.syndrom)), copy.deepcopy(self.toric_code))
            self.loop_check.clear()
            self.states_to_leafnode.clear()
            self.actions_to_leafnode.clear()
            self.actions_to_leafnode_nostring.clear()
        
        #..............................Max Qsa .............................
        actions = self.get_possible_actions(self.syndrom)
        all_Qsa = np.array([[self.Qsa[(s,str(a))] if (s,str(a)) in self.Qsa else 0 for a in position] for position in actions])

        #best action
        maxQ = -float('inf')
        for i in range(len(all_Qsa)):
            for j in all_Qsa[i]:
                if j > maxQ and j !=0:
                    maxQ = j
        index_max = np.where(all_Qsa==maxQ)
        index_max = (index_max[0][0], index_max[1][0])
        best_action1 = actions[index_max[0]][index_max[1]]

        ##### Array of up to 10 of previous actions chosen by MCTS 
        self.last_best_action_array.append(best_action1)
        if len(self.last_best_action_array) == 11:
            del self.last_best_action_array[0] 
        # Check if loop of chosen actions and if so, take next best action and change that to maxQ
        best_action, all_Qsa = self.check_for_loop(all_Qsa, actions, maxQ, index_max, best_action1)
        equal_pos = np.array_equal(best_action.position, best_action1.position)
        equal_act = (best_action.action == best_action1.action)
        equal = equal_pos + equal_act
        if equal != 2:
            self.last_best_action_array[len(self.last_best_action_array)-1] = best_action
        #####

        perspectives = self.generate_perspective(self.args['grid_shift'], self.syndrom)
        perspectives = Perspective(*zip(*perspectives)).perspective

        return all_Qsa, perspectives, actions, best_action, self.last_best_action_array
    

    def search(self, state, actions_taken, root_state, toric_code):
        with torch.no_grad():
            s = str(toric_code.current_state)
            if self.layer == 1 or self.layer == 0:
                reward_multiplier = self.args['reward_multiplier']
            else:
                reward_multiplier = 2
            
            #...........................Check if terminal state.............................
            #if no trivial loop check if terminal state
            terminal_state = toric_code.terminal_state(toric_code.current_state)
            if terminal_state == 0:
                toric_code.eval_ground_state()
                if toric_code.ground_state == True:
                    #print('......Backprop WE WON.......')
                    self.actions_to_leafnode = list(reversed(self.actions_to_leafnode))
                    self.states_to_leafnode = list(reversed(self.states_to_leafnode))
                    v = 100*reward_multiplier
                    self.backpropagation(v)
                elif toric_code.ground_state == False:
                    #print('......Backprop WE LOST.......')
                    self.actions_to_leafnode = list(reversed(self.actions_to_leafnode))
                    self.states_to_leafnode = list(reversed(self.states_to_leafnode))
                    v = -100*reward_multiplier
                    self.backpropagation(v)
                return
            
            #............................Get perspectives ......................
            perspective_list = self.generate_perspective(self.args['grid_shift'], toric_code.current_state)
            perspectives = Perspective(*zip(*perspective_list))
           
            # ............................If leaf node......................................
            if s not in self.Ns:
                self.Ns[s] = 0

                v_from_rollout = self.rollout(perspective_list, copy.deepcopy(toric_code), actions_taken)
                v = v_from_rollout + self.reward*reward_multiplier

                #print('_______')
                #print('Leaf node at layer:',self.layer, ' --> Reward from rollout', v_from_rollout, 'Reward from move:', self.reward*reward_multiplier)
                #print('Leaf node at layer:',self.layer, ' --> v from rollout =',v, 'Reward added to v:', self.reward)
                if s != root_state:
                    #print('......Backprop.......')
                    self.actions_to_leafnode = list(reversed(self.actions_to_leafnode))
                    self.states_to_leafnode = list(reversed(self.states_to_leafnode))
                    self.backpropagation(v)
                else:
                    self.Ns[s] = 1
                return
            else:
                self.reward = 0
                self.selection(s, toric_code, perspectives, perspective_list, actions_taken, root_state)
           
            return

    def selection(self, s, toric_code, perspectives, perspective_list, actions_taken, root_state):
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
  
        #...........................Go down the tree with best action...........................
        a = str(action)
        current_state = copy.deepcopy(toric_code.current_state)
        toric_code.step(action)
        toric_code.current_state = toric_code.next_state 
        next_state = copy.deepcopy(toric_code.current_state)
        reward = self.get_reward(next_state, current_state, toric_code)
        
        
        '''
        print('---->--->---->---->---->---')
        if (s,a) in self.Qsa:
            print('Layer nr:', self.layer, ' Qsa:',self.Qsa[(s,a)], ' Nsa:',self.Nsa[(s,a)], ' ',action, 'Reward for move:',reward, 'L',self.layer,'R',reward)
        else:
            print('Layer nr:', self.layer, ' ',action, 'Reward for move:',reward)
        '''
        self.layer += 1
        self.reward = reward
        self.actions_to_leafnode.append(a)
        self.actions_to_leafnode_nostring.append(action)
        state = toric_code.current_state
        self.search(state, actions_taken, root_state, toric_code)
   
    def backpropagation(self, v):
        self.num_backprop += 1
        i = 0
        for s in self.states_to_leafnode:
            a = self.actions_to_leafnode[i]
            j = i+1
            if (s,a) in self.Qsa:
                Qsa1 = self.Qsa[(s,a)]

                self.Wsa[(s,a)] = self.Wsa[(s,a)] + (self.args['discount_factor']**(j))*v  
                self.Nsa[(s,a)] += 1
                self.Qsa[(s,a)] = self.Wsa[(s,a)]/self.Nsa[(s,a)]

                Qsa2 = self.Qsa[(s,a)]
            else:
                Qsa1 = 0

                self.Wsa[(s,a)] = v
                self.Nsa[(s,a)] = 1
                self.Qsa[(s,a)] = self.Wsa[(s,a)]

                Qsa2 = self.Qsa[(s,a)]
            self.Ns[s] += 1
            i += 1
            self.layer -= 1
            #print('Layer:',self.layer+1,'->',self.layer,' Qsa', Qsa1,'->',Qsa2, '  ',a)   

    def take_random_step(self, toric_code, actions_taken):
        perspective_list = self.generate_perspective(self.args['grid_shift'], toric_code.current_state)
        perspective_index_rand = random.randint(0,len(perspective_list)-1)
        rand_pos = perspective_list[perspective_index_rand].position
        action_index_rand = random.randint(1,3)
        rand_action = Action(np.array(rand_pos), action_index_rand)

        return rand_action

    def rollout(self, perspective_list, toric_code, actions_taken):
        # get reward for every possible action from leaf node           
        actions = self.get_possible_actions(self.syndrom)
        reward = []
        for i in range(len(actions)):
            for j in actions[i]:   
                toric = copy.deepcopy(toric_code)
                current_state = copy.deepcopy(toric.current_state)
                toric.step(j)
                toric.current_state = toric.next_state 
                next_state = copy.deepcopy(toric.current_state)
                
                v = self.get_reward(next_state, current_state, toric)
                reward.append(v)
        v = max(reward) #sum(reward)/len(reward)

        return v

    def get_reward(self, next_state, current_state, toric_code):
        #if no trivial loop check if terminal state
        terminal_state = toric_code.terminal_state(current_state)
        terminal_state = toric_code.terminal_state(toric_code.current_state)
        if terminal_state == 0:
            toric_code.eval_ground_state()
            if toric_code.ground_state == True:
                reward = 1000
            elif toric_code.ground_state == False:
                reward = -1000
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
    
    def get_possible_actions(self, state):
        perspectives = self.generate_perspective(self.args['grid_shift'], state)
        perspectives = Perspective(*zip(*perspectives))
        return [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position] #bytte ut np.array(p_pos)
  
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

    def next_step(self, action):
        self.toric_code.step(action)
        self.toric_code.current_state = self.toric_code.next_state
    
    #### check for loop
    def check_for_loop(self, all_Qsa, actions, maxQ, index_max, best_action):
        action_arr = self.last_best_action_array
        
       #loop, same action repeat right after another 
        if len(action_arr) >= 2:
            equal = self.check_array_loop(action_arr, 2)
            if equal == 2:
                #print('1 loop fix')
                best_action, all_Qsa = self.new_best_action(all_Qsa, actions, maxQ, index_max)
                return best_action, all_Qsa
        #loop of tow actions
        if len(action_arr) >= 4:
            equal = self.check_array_loop(action_arr, 4)
            if equal == 4:
                #print('2 loop fix')
                best_action, all_Qsa = self.new_best_action(all_Qsa, actions, maxQ, index_max)
                return best_action, all_Qsa
        #loop of three actions
        if len(action_arr) >= 6:
            equal = self.check_array_loop(action_arr, 6)
            if equal == 6:
                #print('3 loop fix')
                best_action, all_Qsa = self.new_best_action(all_Qsa, actions, maxQ, index_max)
                return best_action, all_Qsa
         
        #loop of four  actions
        if len(action_arr) >= 8:
            equal = self.check_array_loop(action_arr, 8)
            if equal == 8:
                #print('4 loop fix')
                best_action, all_Qsa = self.new_best_action(all_Qsa, actions, maxQ, index_max)
                return best_action, all_Qsa
         
                #loop of four  actions
        if len(action_arr) >= 10:
            equal = self.check_array_loop(action_arr, 10)
            if equal == 10:
                print('5 loop fix')
                best_action, all_Qsa = self.new_best_action(all_Qsa, actions, maxQ, index_max)
                return best_action, all_Qsa
    
        return best_action, all_Qsa

    #### Check if begining and end of array are equal
    def check_array_loop(self, action_arr, size):
        half_size = int(size/2)
        latest = action_arr[int(len(action_arr)-half_size) : ]
        previous = action_arr[int(len(action_arr)-size) : int(len(action_arr)-half_size)]

        equal = 0
        for i in range(half_size):
            equal_pos = np.array_equal(latest[i].position, previous[i].position)
            equal_act = (latest[i].action == previous[i].action)
            equal += equal_pos + equal_act

        return equal

    #### if in loop get new best action
    def new_best_action(self, all_Qsa, actions, maxQ, index_max):
        #Get second highest Q_value
        second_maxQ = -float('inf')
        for i in range(len(all_Qsa)):
            for j in all_Qsa[i]:
                if j > second_maxQ and j != maxQ and j !=0:
                    second_maxQ = j

        #Get index of all second max values and chose one random amongst them
        index_of_second_maxes = []
        for i in range(len(all_Qsa)):
            n = 0
            for j in all_Qsa[i]:
                if j == second_maxQ:
                    index_of_second_maxes.append((i,n))
                n += 1
        rand_index = index_of_second_maxes[random.randint(0, len(index_of_second_maxes)-1)]              
        best_action = actions[rand_index[0]][rand_index[1]]
        
        # "flip" values in q matrix
        all_Qsa[index_max[0]][index_max[1]] = second_maxQ
        all_Qsa[rand_index[0]][rand_index[1]] = maxQ

        return best_action, all_Qsa



