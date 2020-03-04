import numpy as np
from .util import Action, Perspective, convert_from_np_to_tensor
from .toric_model import Toric_code
import math
import torch
import copy

EPS = 1e-8

class MCTS2():

    def __init__(self, args, nnet, toric_code, device):

        self.toric = toric_code
        self.nnet = nnet
        self.args = args
        self.device = device # 'gpu'
        self.loop_check = set()
        self.current_level = 0 
        self.actions = [] # stores root node actions

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.reward = 0

    def get_policy(self, temp=1):
        
        #...........................get v from search........................

        for i in range(self.args.nr_simulations):
            print("Nr simulation: " + str(i))
            v = self.search(copy.deepcopy(self.toric))
            print('________________________') 
            self.loop_check.clear()
    
        #..............................Policy pi[a|s] .............................

        s = np.array_str(self.toric.current_state)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in self.actions]

        if temp==0:
            bestA = np.argmax(counts)
            pi = [0]*len(counts)
            pi[bestA]=1
            return pi, v, self.actions[bestA]

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        pi = [x/counts_sum for x in counts]
        pi = torch.tensor(pi)
        action = torch.multinomial(pi, 1)        # sample an action according to probabilities probs
        pi = np.array(pi)

        return pi, v, self.actions[action]
        

    def search(self,state):
        #with torch.no_grad():
        print('selection initiated')
        s = str(state.current_state)

        #...........................Check if terminal state.............................

        if np.all(state.current_state == 0):
            if state.eval_ground_state(): #ska det vara self.toric.eval_ground_state(): här istället?
                    #Trivial loop --> gamestate won!
                    return 1
            else:
                #non trivial loop --> game lost!
                return -1

        #..................Get perspectives and batch for network......................

        array_of_perspectives = state.generate_perspective(self.args.gridshift, state.current_state) 
        number_of_perspectives = len(array_of_perspectives) - 1
        perspectives = Perspective(*zip(*array_of_perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        
        # ............................If leaf node......................................

        if s not in self.Ps: 
            self.Ps[s], v = self.expansion(batch_perspectives, s)
            return v

        # ..........................Get best action...................................
        
        cur_best = -float('inf')
        best_action = -1
        self.current_level += 1

        for perspective in range(number_of_perspectives):
            for action in range(1, 4):

                a = Action(batch_position_actions[perspective], action)
                probability_matrix = self.Ps[s][perspective][action-1]

                if self.current_level == 1:
                    self.actions.append(a)

                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*probability_matrix.data.numpy()*\
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*probability_matrix.data.numpy()*math.sqrt(self.Ns[s] + EPS)
                # loop_check to make sure the same bits are not flipped back and forth, creating an infinite recursion loop
                if u > cur_best and (s,a) not in self.loop_check:
                    cur_best = u
                    best_action = a

        self.loop_check.add((s,best_action))
        a = best_action
        state.step(a)
        self.reward = self.get_reward(state)
        state.current_state = state.next_state

        print('-------')
        v = self.search(copy.deepcopy(state))
        print('## Time for backprop ##')

        # ............................BACKPROPAGATION................................
        
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Qsa[(s,a)] +self.reward + self.args.disscount_factor*v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
            print('Adding!')
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
            print('First Time!')

        self.Ns[s] += 1
        return v
    
    # Reward
    def get_reward(self, state):
        terminal = np.all(state.next_state==0)
        if terminal == True:
            reward = 100
            print('Reward = ', reward)
        else:
            defects_state = np.sum(state.current_state)
            defects_next_state = np.sum(state.next_state)
            reward = defects_state - defects_next_state
            print('Reward = ', reward)

        return reward
    
    # Expansion from leaf node
    def expansion(self, batch_perspectives, s):
        self.Ps[s] = self.nnet.forward(batch_perspectives)  # matris med q,värdena för de olika actionen för alla olika perspektiv i en batch
        v = torch.max(self.Ps[s])
        v = v.data.numpy()
        # Normalisera
        sum_Ps_s = torch.sum(self.Ps[s]) 
        self.Ps[s] = self.Ps[s]/sum_Ps_s    
            
        # ej besökt detta state tidigare sätter dessa parametrar till 0
        self.Ns[s] = 0
        
        return self.Ps[s], v

    
    def rollout(self, batch_perspectives, s):
        pass


  



