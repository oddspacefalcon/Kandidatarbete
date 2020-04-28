import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def predict(plot_range, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode):
    # Generate folder structure, all results are stored in the data_results folder 
    timestamp = time.strftime("%y_%m_%d__%H__%M__%S__")
    PATH = 'Result_Time_Test/prediction__' +str(NETWORK_FILE_NAME) +'__'+ timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    # Path for the network to use for the prediction
    PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'  
    data_result = np.zeros((1, 3))
    data_win_rate = np.zeros((1, 2))

    for i in range(plot_range):
        i += 1
        prediction_list_p_error = [i*0.01]
        start = time.time()
        # initialize RL class
        rl = RL(Network=network,
                Network_name=NETWORK_FILE_NAME,
                system_size=system_size,
                device=device)
        
        # generate syndrome with a random amount of errors 
        minimum_nbr_of_qubit_errors = 0 
    
        print('Prediction')
        error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, solve_time, avarage_nr_steps = rl.prediction(
            num_of_predictions=num_of_predictions, 
            num_of_steps=75, 
            PATH=PATH2, 
            prediction_list_p_error=prediction_list_p_error,
            minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
            plot_one_episode=plot_one_episode) # FÖR att plotta
        
        # runtime of prediction
        runtime = time.time()-start
        runtime = runtime / 3600
        
        avarage_solve_time = np.sum(solve_time)/len(solve_time)
        avarage_steps = np.sum(avarage_nr_steps)/len(avarage_nr_steps)
        win_rate = (num_of_predictions-(len(failed_syndroms)/2))/num_of_predictions
        print(avarage_solve_time, 'avarage solve time')
        print(avarage_steps,'avarage nr of steps')
        print(ground_state_list, 'ground state conserved rate.')
        print(average_number_of_steps_list, 'average number of steps.')
        print(len(failed_syndroms)/2, 'failed syndroms out of',num_of_predictions, 'predictions --> Win rate:',win_rate)
        
        # save training settings in txt file 
       
        data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], avarage_solve_time, avarage_steps]]), axis= 0)
        np.savetxt(PATH + '/data_result_solve_time.txt', data_result, header='# P_error, Avarage solve time, Avarage steps', delimiter=',', fmt="%s")

        data_win_rate = np.append(data_win_rate ,np.array([[prediction_list_p_error[0], win_rate]]), axis= 0)
        np.savetxt(PATH + '/data_result_win_rate.txt', data_win_rate, header='# P_error, Win rate', delimiter=',', fmt="%s")

    return PATH


#######################################
device = 'cuda'
system_size = 5
network = NN_11
NETWORK_FILE_NAME = 'MCTS_size_5_NN_11_epoch_79'                                                             
num_of_predictions = 4000
plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
predict(plot_range, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)

########################################

#######################################
device = 'cuda'
system_size = 9
network = ResNet18
NETWORK_FILE_NAME = 'MCTS_size_9_NN_11_epoch_279'                                                             
num_of_predictions = 4000
plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
predict(plot_range, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)

########################################

#######################################
device = 'cuda'
system_size = 13
network = ResNet18
NETWORK_FILE_NAME = 'MCTS_Size_13_ResNet18_steps_15800_trained_to_P_0.09'                                                             
num_of_predictions = 4000
plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
predict(plot_range, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)

########################################



