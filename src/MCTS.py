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
    Ett objekt representerar ett MCTS-träd
    """

    def __init__(self, toric_code, model, device, args):
        self.toric_code = toric_code # toric_model-objekt
        self.model = model  # resnet
        self.args = args    # c_puct, num_simulations (antalet noder), grid_shift 
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.device = device # 'cpu'
        self.actions = []
        self.current_level = 0
        self.last_move = None

    def search(self, state):
        # np.arrays är unhashable, behöver string
        s = np.array_str(state.current_state)

        perspectives = state.generate_perspective(self.args['grid_shift'], state.current_state) # genererar väl inte olika olika ggr???
        number_of_perspectives = len(perspectives) - 1
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        if np.all(state.current_state == 0):
            # if terminal state
            # Davids rewardsystem / number_of_perspectives. Ett försök att få samma skala i UCB som alphago. Tål att tänkas mer på
            return 100 / number_of_perspectives
            #return 1 # terminal <=> vunnit

        if s not in self.Ps:
            # leaf node => expand
            self.Ps[s] = self.model.forward(batch_perspectives)

            self.Ns[s] = 0

            #return random.random()
            
            E_t = np.sum(state.last_state)
            E_t1 = np.sum(state.current_state)
            # Davids rewardsystem / number_of_perspectives. Ett försök att få samma skala i UCB som alphago. Tål att tänkas mer på
            #print((E_t - E_t1) / number_of_perspectives)
            return (E_t - E_t1) / number_of_perspectives

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound, using all perspectives of toric code s

        self.current_level += 1

        for perspective in range(number_of_perspectives):
            for action in range(1, 4):

                a = Action(batch_position_actions[perspective], action)

                if self.current_level == 1:
                    self.actions.append(a)

                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['cpuct']*self.Ps[s][perspective][action-1]*\
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args['cpuct']*self.Ps[s][perspective][action-1]*math.sqrt(self.Ns[s] + EPS)
                                
                if u > cur_best and a != self.last_move:
                    cur_best = u
                    best_act = a

        self.last_move = best_act
        a = best_act
        state.step(a)
        state.last_state = state.current_state
        state.current_state = state.next_state

        v = self.search(copy.deepcopy(state))

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v

    def get_probs_actions(self, temp=1):

        s = np.array_str(self.toric_code.current_state)

        for i in range(self.args['num_simulations']):
             # om inte kopia så sparas alla operationer man gör på toric koden mellan varje iteration
            self.search(copy.deepcopy(self.toric_code))
        
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in self.actions]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs, self.actions

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        probs = torch.tensor(probs)
        probs = probs.view(-1, 3)
        return probs, self.actions