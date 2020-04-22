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
    PATH = 'Results/prediction__' +str(NETWORK_FILE_NAME) +'__'+ timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    # Path for the network to use for the prediction
    PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'  
    data_result = np.zeros((1, 3))
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
        print(avarage_solve_time)
       
        data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], avarage_solve_time, avarage_steps]]), axis= 0)
        np.savetxt(PATH + '/data_result_solve_time.txt', data_result, delimiter=',', fmt="%s")
    return PATH

def plot(PATH, plot_range, system_size):
    with open(PATH + '\data_result_solve_time.txt')as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    time = []
    avarage_nr_steps = []
    P_error= []
    II =  []
    i = 0
    for row in  content:    
        P_e, Time, steps = row.split(',')
        P_e = float(P_e)
        Time = float(Time)
        steps = float(steps)

        if P_e != 0.0:
            P_error.append(P_e)
            time.append(Time)
            avarage_nr_steps.append(steps)
            II.append(i)
            i = i+1
    print(avarage_nr_steps)
    
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Tid [s]', fontsize=24)
    ax1.set_xlabel('$P_e$', fontsize=24)
    #ax1.scatter(P_error, time, label='d = '+str(system_size), color='steelblue', marker='o')
    lns1 = ax1.plot(P_error,time, color='steelblue', label='Time')
    ax1.set_xlim(0.005,plot_range*0.01+0.005)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Medelantal drag', fontsize=24)
    lns2 = ax2.plot(P_error,avarage_nr_steps, alpha = 0, label='Steg')

    # added these three lines
    #lns = lns1+lns2
    #labs = [l.get_label() for l in lns]
    #plt.legend(lns, labs, loc=0, fontsize = 24)
    ax1.xaxis.set_tick_params(labelsize=24)
    ax1.yaxis.set_tick_params(labelsize=24)
    ax2.yaxis.set_tick_params(labelsize=24)
    plt.title('Prestation vid lyckad felkorrigering', fontsize=24)
    plt.savefig(PATH+'/Result_plot'+'.png')
    plt.show()


#######################################
device = 'cuda'
system_size = 5
network = ResNet18 # Valid network names: NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
#NETWORK_FILE_NAME = 'Main_Size_5_NN_11_steps_7000' # this file is stored in the network folder and contains the trained agent.
NETWORK_FILE_NAME = 'Main_Size_5_NN_11_steps_7000'                                                             
num_of_predictions = 2000
plot_range = 10 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
########################################
#PATH = predict(plot_range, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)
PATH  = 'Results/Main_Size_5_NN_11_steps_epoch_7_steps_7000__20_04_15__11__54__14__'
plot(PATH, plot_range, system_size)



