import numpy as np
import math
import os
import time

from src.toric_model import Toric_code, Action, Perspective
import torch
import _pickle as cPickle
from src.RL import RL
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##########################################################################
# common system sizes are 3,5,7 and 9, grid size must be odd!
system_size = 5
# valid network names:
#   NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
network = ResNet18
# this file is stored in the network folder and contains the trained agent.
NETWORK_FILE_NAME = 'Size_7_NN_17'
num_of_predictions = 1

# initialize RL class
rl = RL(Network=network,
        Network_name=NETWORK_FILE_NAME,
        system_size=system_size,
        device=device)

# initial syndrome error generation
# generate syndrome with error probability 0.1
prediction_list_p_error = [0.1]
# generate syndrome with a fixed amount of errors
# minimum number of erorrs for logical qubit flip
minimum_nbr_of_qubit_errors = int(system_size/2)+1

# Generate folder structure, all results are stored in the data folder
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/prediction__' + str(NETWORK_FILE_NAME) + '__' + timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)

def run_MCTS():

    system_size = 3
    nr_errors = 2
    toric = Toric_code(system_size)
    toric.generate_n_random_errors(nr_errors)

    MCTS = MCTS()





class MCTS():

    
   # initiera mcts
    def __init__(self):

        self.num_sim = 25 # antal itterationer

      
        self.Q_sa = {}      # stores Q values for s,a
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited (root node)
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game game ended for board s
        # stores game valid moves for board s (all available branches from this state)
        self.Vs = {}
    

    def get_current_state(self):
        current_state = toric.current_state
        return current_state


    def ask_Network(self, state):
        '''
        Fråga network efter p(s,a), och v(s).
        '''
        pass

    def search(self, position):
        pass

    def Expand(self):
        pass

    def backprop(self):
        pass

    def MakeMove(self):
        pass


  
