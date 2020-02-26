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
        if(toric_code == None):
            if(syndrom == None):
                raise ValueError("Invalid imput: toric_code or syndrom cannot both have a None value")
            else:
                self.syndrom = syndrom
                self.toric_code = None
        else:
            self.toric_code = toric_code
            self.syndrom = self.toric_code.current_state

        self.toric_code = toric_code # toric_model object
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

    def search(self, state):
        # np.arrays unhashable, needs string
        s = np.array_str(state.current_state)

        perspectives = state.generate_perspective(self.args['grid_shift'], state.current_state)
        number_of_perspectives = len(perspectives)
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        if np.all(state.current_state == 0):
            # if terminal state
            return 1 # terminal <=> vunnit

        if s not in self.Ps:
            # leaf node => expand
            self.Ps[s], v = self.model.forward(batch_perspectives)
            self.Ns[s] = 0
            return v

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
                # loop_check to make sure the same bits are not flipped back and forth, creating an infinite recursion loop
                if u > cur_best and (s,a) not in self.loop_check:
                    cur_best = u
                    best_act = a

        self.loop_check.add((s,best_act))

        a = best_act
        state.step(a)
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

    def get_probs_action(self, temp=1):

        s = np.array_str(self.toric_code.current_state)

        for i in range(self.args['num_simulations']):
             # if not copied the operations on the toric code would be saved over every tree
             #instead add toric_codes syndrom
            self.search(copy.deepcopy(self.toric_code))
             # clear loop_check so the same path can be taken in new tree
            self.loop_check.clear()

        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in self.actions]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs, self.actions[bestA]

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        probs = torch.tensor(probs)
        # sample an action according to probabilities probs
        action = torch.multinomial(probs, 1)
        probs = probs.view(-1, 3)
        return probs, self.actions[action]