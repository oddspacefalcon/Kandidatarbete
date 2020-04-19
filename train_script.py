import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

##########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = 'cpu'

# valid network names: 
#   NN_11
#   NN_17
#   ResNet18
#   ResNet34
#   ResNet50
#   ResNet101
#   ResNet152
NETWORK = NN_11

# common system sizes are 3,5,7 and 9 
# grid size must be odd! 
SYSTEM_SIZE = 5

# For continuing the training of an agent
continue_training = True
# this file is stored in the network folder and contains the trained agent.  
NETWORK_FILE_NAME = 'size_5_size_5_NN_11_epoch_14_memory_proportional_optimizer_Adam__steps_14000_learning_rate_0.00025'

# initialize RL class and training parameters 
rl = RL(Network=NETWORK,
        Network_name=NETWORK_FILE_NAME,
        system_size=SYSTEM_SIZE,
        p_error=0.1,
        replay_memory_capacity=1000, 
        learning_rate=0.00025,
        max_nbr_actions_per_episode=50,
        device=device,
        replay_memory='proportional',  # proportional or uniform
        num_simulations=100, 
        discount_factor=0.95, 
        epsilon=0.1, 
        memory_reset=150)


# generate folder structure 
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/training__' +str(NETWORK_FILE_NAME) +'_'+str(SYSTEM_SIZE)+'__' + timestamp
PATH_epoch = PATH + '/network_epoch'
if not os.path.exists(PATH):
    os.makedirs(PATH)
    os.makedirs(PATH_epoch)

# load the network for continue training 
if continue_training == True:
    print('continue training')
    PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
    rl.load_network(PATH2)

# train for n epochs the agent (test parameters)
rl.train_for_n_epochs(training_steps=1000,
                    num_of_predictions=200,
                    num_of_steps_prediction=50,
                    epochs=50,
                    optimizer='Adam',
                    batch_size=32,
                    directory_path = PATH,
                    prediction_list_p_error=[0.1],
                    replay_start_size=50)


"""rl.train_for_n_epochs(training_steps=10000,
                            num_of_predictions=100,
                            epochs=100,
                            optimizer='Adam',
                            batch_size=32,
                            directory_path = PATH,
                            prediction_list_p_error=[0.1],
                            minimum_nbr_of_qubit_errors=0"""