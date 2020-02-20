import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
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


    def search(self, state):
        # np.arrays är unhasable, behöver string
        s = copy.deepcopy(np.array_str(self.toric_code.current_state))

        if np.all(state.current_state == 0):
            # if terminal state
            return 1 # terminal <=> vunnit

        perspectives = state.generate_perspective(self.args['grid_shift'], state.current_state)
        number_of_perspectives = len(perspectives) - 1
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        if s not in self.Ps:
            # leaf node => expand
            self.Ps[s] = self.model.forward(batch_perspectives)

            # defects_state = np.sum(self.perspective)
            # defects_next_state = np.sum(self.toric.next_state)
            # v = defects_state - defects_next_state

            self.Ns[s] = 0

            # tills resnet är fixad
            v = 0
            return v


        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound, using all perspectives of toric code s

        for perspective in range(number_of_perspectives):
            for action in range(1, 4):
                a = Action(batch_position_actions[perspective], action)

                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['cpuct']*self.Ps[s][perspective][action-1]*\
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args['cpuct']*self.Ps[s][perspective][action-1]*math.sqrt(self.Ns[s] + EPS)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a

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


    def get_probs_values(self, temp=1):

        s = copy.deepcopy(np.array_str(self.toric_code.current_state))

        for i in range(self.args['num_simulations']):
            v = self.search(self.toric_code)

        perspectives = self.toric_code.generate_perspective(self.args['grid_shift'], self.toric_code.current_state)
        number_of_perspectives = len(perspectives) - 1
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position

        actions = [Action(batch_position_actions[perspective], action) for perspective \
             in range(number_of_perspectives) for action in range(1, 4)]
        
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in actions]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs, v

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs, v