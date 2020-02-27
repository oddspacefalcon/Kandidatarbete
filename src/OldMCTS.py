import numpy as np
from util import Action, Perspective, convert_from_np_to_tensor
from toric_model import Toric_code
import math
import torch
import copy

EPS = 1e-8

class MCTS():

    def __init__(self, args, nnet, toric_code, device):

        self.toric = toric_code
        self.nnet = nnet
        self.args = args
        self.root_state = copy.deepcopy(self.toric)
        self.actions_taken = []
        self.device = device # 'gpu'
        self.batch_perspectives_temp = np.zeros(1)
        self.loop_check = set()
        self.current_level = 0 
        self.actions = [] # stores root node actions

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s


    def get_policy(self, temp=0):
        
        #...........................get v from search........................

        for i in range(self.args.nr_simulations):
            print("Nr simulation: " + str(i))
            v = self.search(copy.deepcopy(self.toric))
            print('________________________') 
            self.loop_check.clear()
    

        #..............................Policy pi[a|s] .............................

        # OBS: borde få lista med a från search också. Bör fixa
        self.toric.qubit_matrix = self.root_state
        s = str(self.root_state)

        array_of_perspectives = self.toric.generate_perspective(self.args.gridshift//2, self.toric.current_state) 
        perspective_pos = Perspective(*zip(*array_of_perspectives)).position
        actions = [[str(Action(p_pos, x)) for x in range(3)] for p_pos in perspective_pos]
        pi = [[self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in perspective] for perspective in actions]

        if temp==0:
            bestA = np.argmax(pi)
            pi = [0]*len(pi)
            pi[bestA]=1
            return pi, v

        #pi = [[prob**(1./temp) for prob in perspective] for perspective in pi]
        pi = torch.tensor(pi)
        action = torch.multinomial(pi, 1)
        #pi = np.array(pi)

        # return flat prob matrix
        return pi, self.actions[action]

    
    def search(self,state):
        print('selection initiated')
        s = str(state.current_state)

        #...........................Check if terminal state.............................

        #obs self.Es[s] får skalärvärde 0,1,-1. Kollar om vi är i ett terminal state av ngt slag
        if s not in self.Es:    
            if self.toric.terminal_state('next_state') == 1:  #then not terminal state
                print('Not in terminal state')
                self.Es[s] = 0
            else:
                if self.toric.eval_ground_sate():
                    #Trivial loop --> gamestate won!
                    self.Es[s] = 1 
                    print('We Won! :)')
                    
                else:
                    #non trivial loop --> game lost!
                    self.Es[s] = - 1 
                    print('Lost :(')
            # returnera om vi vunnti (1) eller förlorat (-1)
            if self.Es[s] != 0:
                v = self.Es[s]
                return v #returnerar -1 eller 1


        #..................Get perspectives and batch for network......................

        array_of_perspectives = state.generate_perspective(self.args.gridshift, state.current_state) 
        number_of_perspectives = len(array_of_perspectives) - 1
        perspectives = Perspective(*zip(*array_of_perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        
        # ............................If leaf node......................................

        # batch_perspective biten pga förhindra ett error som dök upp till och från nu under testfasen...
        if s not in self.Ps: # or self.batch_perspectives_temp.any() != np.array(perspectives.perspective).any(): 

            self.Ps[s], v = self.expansion(batch_perspectives, s)
            print('Leaf node, asking network: shape(Ps)=', self.Ps[s].data.numpy().shape, '  v=', v)
            print(batch_perspectives.data.numpy().shape)
            #print(self.Ps)

            return v
        #self.batch_perspectives_temp = np.array(perspectives.perspective)

        # ..........................Get best action...................................
            
        #positionen blir här en tuple
        actions = [[Action(p_pos, x) for x in range(3)] for p_pos in batch_position_actions]
        UpperConfidence = self.UCBpuct(self.Ps[s], actions, s)
        perspective_index, action_index = np.unravel_index(np.argmax(UpperConfidence), UpperConfidence.shape)
        best_perspective = array_of_perspectives[perspective_index]

        best_action = Action(best_perspective.position, action_index+1)
        a = str(best_action)
        self.actions_taken.append((s,a))
        
        # ............................BACKPROPAGATION................................

        if (s,a) in self.Nsa:
            self.Nsa[(s,a)] +=1
            print('Adding: Nsa = ', self.Nsa[(s,a)])
        else:
            self.Nsa[(s,a)] = 1
            print('First time: Nsa = ', self.Nsa[(s,a)])
        
        if s in self.Ns:
            self.Ns[s]+=1
            print('Adding: Ns = ', self.Ns[s])
        else:
            self.Ns[s]=1
            print('First time: Ns = ', self.Ns[s])

        print('HEEEEEEEEEEEEEJ')
        state.step(best_action)
        state.current_state = state.next_state
        v = self.search(copy.deepcopy(state)) #np.copy(self.toric.qubit_matrix)
        
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = ((self.Nsa[(s,a)]-1)*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)])
            print('Updating: Qsa = ', self.Qsa[(s,a)])
        else:
            self.Qsa[(s,a)] = v
            print('First time: Qsa = ', self.Qsa[(s,a)])


        return v
        



    # At leaf node
    def expansion(self, batch_perspectives, s):
        self.Ps[s], v = self.nnet.forward(batch_perspectives)  # matris med q,värdena för de olika actionen för alla olika perspektiv i en batch
        
        # Normalisera
        sum_Ps_s = torch.sum(self.Ps[s]) 
        self.Ps[s] = self.Ps[s]/sum_Ps_s    

        # Gör så att v blir en skalär [-1,1] (medelvärdet för alla perspektiven)
        sum_v = torch.sum(v) 
        v = sum_v/(v.size(0))
        v = v.data.numpy()
            
        # ej besökt detta state tidigare sätter dessa parametrar till 0
        self.Ns[s] = 0
        
        return self.Ps[s], v

        
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

        return current_Qsa + self.args.cpuct*probability_matrix.data.numpy()*np.sqrt(current_Ns/(1+current_Nsa))

    
    def rollout(self, batch_perspectives, s):
        pass


  



