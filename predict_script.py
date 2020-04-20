import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

start = time.time()

device = 'cuda'

##########################################################################

# common system sizes are 3,5,7 and 9. Grid size must be odd! 
system_size = 9

# valid network names: NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
network = ResNet18

# this file is stored in the network folder and contains the trained agent.  
#NETWORK_FILE_NAME = 'Size_5_NN_11' #Win rate 0.6999, P=0.15
NETWORK_FILE_NAME = 'size_9_Size_9_ResNet18_epoch_9_memory_uniform_optimizer_Adam__steps_900_learning_rate_0.00025' #Win rate: 0.961, P=0.05
#NETWORK_FILE_NAME = 'Size_5_NN_11_steps_7150' #Win rate: 0.964, P=0.05

# d = 11

#NETWORK_FILE_NAME ='Size_5_NN_11'     # Main P_e=0.05 ->0.9535 __ P_e=0.03 ->0.9895 __ P_e=0.1 ->0.775__ P_e=0.15 ->0.5
num_of_predictions = 100

# initialize RL class
rl = RL(Network=network,
        Network_name=NETWORK_FILE_NAME,
        system_size=system_size,
        device=device)

# initial syndrome error generation 
# generate syndrome with error probability 0.1 
prediction_list_p_error = [0.01]
# generate syndrome with a fixed amount of errors 
minimum_nbr_of_qubit_errors = 0 #int(system_size/2)+1 # minimum number of erorrs for logical qubit flip

# Generate folder structure, all results are stored in the data folder 
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'Results/prediction__' +str(NETWORK_FILE_NAME) +'__'+  timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)

# Path for the network to use for the prediction
PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
print('Prediction')
error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = rl.prediction(
    num_of_predictions=num_of_predictions, 
    num_of_steps=75, 
    PATH=PATH2, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=False) # FÖR att plotta

# runtime of prediction
runtime = time.time()-start
runtime = runtime / 3600

win_rate = (num_of_predictions-(len(failed_syndroms)/2))/num_of_predictions
print(ground_state_list, 'ground state conserved rate.')
print(error_corrected_list, 'error corrected list')
print(average_number_of_steps_list, 'average number of steps.')
print(len(failed_syndroms)/2, 'failed syndroms out of',num_of_predictions, 'predictions --> Win rate:',win_rate)



  
# save training settings in txt file 
data_all = np.array([[NETWORK_FILE_NAME, num_of_predictions, error_corrected_list[0], ground_state_list[0],average_number_of_steps_list[0], len(failed_syndroms)/2, runtime, prediction_list_p_error[0], win_rate ]])
data_result = np.array([[prediction_list_p_error[0], win_rate]])
np.savetxt(PATH + '/data_all.txt', data_all, header='network, failure_rate, error corrected, ground state conserved, average number of steps, mean q value, number of failed syndroms, runtime (h), Prob error, Win Rate', delimiter=',', fmt="%s")
np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%s")

