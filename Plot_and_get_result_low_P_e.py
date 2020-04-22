<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def predict(plot_range, predict_nr, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode):
    # Generate folder structure, all results are stored in the data_results folder 
    timestamp = time.strftime("%y_%m_%d__%H__%M__%S__")
    PATH = 'drive/My Drive/Colab Notebooks/Results/prediction__' +str(NETWORK_FILE_NAME) +'__'+ timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    # Path for the network to use for the prediction
    PATH2 = 'drive/My Drive/Colab Notebooks/network/'+str(NETWORK_FILE_NAME)+'.pt'  
    data_result = np.zeros((1, 2))
    for i in range(1):
        i += 1
        #print(i)
        #if i != 7:
            #print(i)
            #continue
        #print('i:',i)
            
        prediction_list_p_error = [0.001]
        start = time.time()
        # initialize RL class
        rl = RL(Network=network,
                Network_name=NETWORK_FILE_NAME,
                system_size=system_size,
                device=device)
        
        
        # generate syndrome with a random amount of errors 
        minimum_nbr_of_qubit_errors = 0 
    
        print('Next')
        error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = rl.prediction(
            num_of_predictions=num_of_predictions, 
            num_of_steps=75, 
             PATH=PATH2, 
             prediction_list_p_error=prediction_list_p_error,
             minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
             plot_one_episode=plot_one_episode) # FÖR att plotta
        
         # runtime of prediction
        runtime = time.time()-start
        runtime = runtime / 3600
        
        fail_rate = (len(failed_syndroms)/2)/num_of_predictions
        win_rate = (num_of_predictions-(len(failed_syndroms)/2))/num_of_predictions
            #print(ground_state_list, 'ground state conserved rate.')
            #print(fail_rate, 'fail_rate')
            #print(average_number_of_steps_list, 'average number of steps.')
            #print(len(failed_syndroms)/2, 'failed syndroms out of',num_of_predictions, 'predictions --> Win rate:',win_rate)
        
            # save training settings in txt file 
            #data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], win_rate]]), axis= 0)
            #np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%s")

        data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], fail_rate]]), axis= 0)
        np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%s")
    return PATH

def plot(PATH, plot_range, system_size):
    with open(PATH + '/data_result.txt')as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    P_success = []
    P_error= []
    II =  []
    i = 0
    for row in  content:    
        P_e, P_s = row.split(',')
        P_e = float(P_e)
        P_s = float(P_s)
        if P_e != 0.0:
            P_error.append(P_e)
            P_success.append(P_s)
            II.append(i)
            i = i+1
    
    fig, ax = plt.subplots()
    ax.scatter(P_error, P_success,label='d = '+str(system_size), color='steelblue', marker='o')
    ax.legend(fontsize = 24)
    ax.plot(P_error,P_success, color='steelblue')
    ax.set_xlim(10**(-3)-0.0001,10**(-2)+0.001)
    ax.set_ylim(2*10**(-6),10**(-3)+0.0001)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('$P_e$', fontsize=24)
    plt.ylabel('$P_f$', fontsize=24)
    plt.title('Prestation vid låga $P_e$', fontsize=24)

    plt.tick_params(axis='both', labelsize=24)
    #fig.set_figwidth(10)
    plt.savefig(PATH+'/Result_plot'+'.png')
    plt.show()


#######################################
device = 'cuda'
system_size = 5
predict_nr = 1
network = ResNet18 # Valid network names: NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
NETWORK_FILE_NAME = 'Main_Size_5_NN_11_steps_7000' # this file is stored in the network folder and contains the trained agent.                                                             
num_of_predictions = 1000000
plot_range = 100 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
########################################
#PATH = predict(plot_range, predict_nr, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)
PATH  = 'Results/Fail_rate_low_P_error__Main_Size_5_NN_11_steps_7000__20_04_19__21__55__54__'
plot(PATH, plot_range, system_size)




=======
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def predict(plot_range, predict_nr, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode):
    # Generate folder structure, all results are stored in the data_results folder 
    timestamp = time.strftime("%y_%m_%d__%H__%M__%S__")
    PATH = 'drive/My Drive/Colab Notebooks/Results/prediction__' +str(NETWORK_FILE_NAME) +'__'+ timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    # Path for the network to use for the prediction
    PATH2 = 'drive/My Drive/Colab Notebooks/network/'+str(NETWORK_FILE_NAME)+'.pt'  
    data_result = np.zeros((1, 2))
    for i in range(1):
        i += 1
        #print(i)
        #if i != 7:
            #print(i)
            #continue
        #print('i:',i)
            
        prediction_list_p_error = [0.001]
        start = time.time()
        # initialize RL class
        rl = RL(Network=network,
                Network_name=NETWORK_FILE_NAME,
                system_size=system_size,
                device=device)
        
        
        # generate syndrome with a random amount of errors 
        minimum_nbr_of_qubit_errors = 0 
    
        print('Next')
        error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = rl.prediction(
            num_of_predictions=num_of_predictions, 
            num_of_steps=75, 
             PATH=PATH2, 
             prediction_list_p_error=prediction_list_p_error,
             minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
             plot_one_episode=plot_one_episode) # FÖR att plotta
        
         # runtime of prediction
        runtime = time.time()-start
        runtime = runtime / 3600
        
        fail_rate = (len(failed_syndroms)/2)/num_of_predictions
        win_rate = (num_of_predictions-(len(failed_syndroms)/2))/num_of_predictions
            #print(ground_state_list, 'ground state conserved rate.')
            #print(fail_rate, 'fail_rate')
            #print(average_number_of_steps_list, 'average number of steps.')
            #print(len(failed_syndroms)/2, 'failed syndroms out of',num_of_predictions, 'predictions --> Win rate:',win_rate)
        
            # save training settings in txt file 
            #data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], win_rate]]), axis= 0)
            #np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%s")

        data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], fail_rate]]), axis= 0)
        np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%s")
    return PATH

def plot(PATH, plot_range, system_size):
    with open(PATH + '/data_result.txt')as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    P_success = []
    P_error= []
    II =  []
    i = 0
    for row in  content:    
        P_e, P_s = row.split(',')
        P_e = float(P_e)
        P_s = float(P_s)
        if P_e != 0.0:
            P_error.append(P_e)
            P_success.append(P_s)
            II.append(i)
            i = i+1
    
    fig, ax = plt.subplots()
    ax.scatter(P_error, P_success,label='d = '+str(system_size), color='steelblue', marker='o')
    ax.legend(fontsize = 13)
    ax.plot(P_error,P_success, color='steelblue')
    ax.set_xlim(10**(-3),10**(-2)+0.0001)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('$P_e$', fontsize=14)
    plt.ylabel('$P_f$', fontsize=14)

    plt.tick_params(axis='both', labelsize=15)
    #fig.set_figwidth(10)
    plt.savefig(PATH+'/Result_plot'+'.png')
    plt.show()


#######################################
device = 'cuda'
system_size = 5
predict_nr = 1
network = ResNet18 # Valid network names: NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
NETWORK_FILE_NAME = 'Main_Size_5_NN_11_steps_7000' # this file is stored in the network folder and contains the trained agent.                                                             
num_of_predictions = 1000000
plot_range = 100 # plot from P_error = 0.01 to plot_range*0.01
plot_one_episode = False
########################################
#PATH = predict(plot_range, predict_nr, device, system_size, network, NETWORK_FILE_NAME, num_of_predictions, plot_one_episode)
PATH  = 'Results/Fail_rate_low_P_error__Main_Size_5_NN_11_steps_7000__20_04_19__21__55__54__'
plot(PATH, plot_range, system_size)




>>>>>>> ebae4798751da532ef294496a220bd6aeb908d8c
