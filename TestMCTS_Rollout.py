import numpy as np
import time
import os
import torch
import copy
import _pickle as cPickle
from src.RL import RL
import matplotlib.pyplot as plt

from src.MCTS_Rollout2 import MCTS_Rollout2
from src.MCTS_Rollout import MCTS_Rollout


from src.toric_model import Toric_code
from src.toric_model import Action
from src.toric_model import Perspective

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.util import incremental_mean, convert_from_np_to_tensor, Transition, Action, Qval_Perspective

##########################################################################


def predictionMCTS(args,num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, cpuct=0.0,
    show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, print_Q_values=False, save_prediction=False,
    system_size=3):

    grid_shift = system_size//2
    # init matrices 
    ground_state_list = np.zeros(len(prediction_list_p_error))
    error_corrected_list = np.zeros(len(prediction_list_p_error))
    average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
    mean_q_list = np.zeros(len(prediction_list_p_error))
    failed_syndroms = []
    nr_failed_syndroms = 0
    # loop through different p_error
    for i, p_error in enumerate(prediction_list_p_error):
        ground_state = np.ones(num_of_predictions, dtype=bool)
        error_corrected = np.zeros(num_of_predictions)
        mean_steps_per_p_error = 0
        mean_q_per_p_error = 0
        steps_counter = 0
        for j in range(num_of_predictions):
            num_of_steps_per_episode = 0
            prev_action = 0
            terminal_state = 0
            last_best_action = None
            last_best_action_array = []
            
            print('prediction nr', j)

            # generate random syndrom
            toric = Toric_code(system_size)
            #toric.generate_random_error(p_error)
            
            if minimum_nbr_of_qubit_errors == 0:
                toric.generate_random_error(p_error)
            else:
                toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
            
            terminal_state = toric.terminal_state(toric.current_state)
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                toric.plot_toric_code(toric.current_state, 'initial_syndrom')
            
            init_qubit_state = copy.deepcopy(toric.qubit_matrix)
            mcts = MCTS_Rollout2('cpu', args, copy.deepcopy(toric), None) #Classic rollout MCTS
            # solve syndrome
            while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1

                # choose MCTS action
                 
                _,_,_,action  = mcts.get_qs_actions()

                # choose MCTS action
                #mcts = MCTS_Rollout('cpu', args, copy.deepcopy(toric), None, last_best_action) #Classic rollout MCTS 
                #_,_,_,action , last_best_action = mcts.get_qs_actions()

                
                print('___________________________________________________')
                print(toric.current_state)
                print('-----------')
                print('best action', action)
                add1 = np.sum(toric.current_state)
                
     
                prev_action = action
                toric.step(action)
                toric.current_state = toric.next_state
                terminal_state = toric.terminal_state(toric.current_state)
                mcts.next_step(action)
                
                
                print(toric.current_state)
                add2 = np.sum(toric.current_state)
                print('sum diff: ', add1-add2)
                print('Tot err left: ', add2)
                

                if plot_one_episode == True and j == 0 and i == 0:
                    toric.plot_toric_code(toric.current_state, 'step_'+str(num_of_steps_per_episode))
                #time.sleep(10)

            # compute mean steps 
            mean_steps_per_p_error = incremental_mean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
            # save error corrected 
            error_corrected[j] = toric.terminal_state(toric.current_state) # 0: error corrected # 1: error not corrected    
            # update groundstate
            toric.eval_ground_state()                                                          
            ground_state[j] = toric.ground_state # False non trivial loops

            if terminal_state == 1 or toric.ground_state == False:
                failed_syndroms.append(init_qubit_state)
                failed_syndroms.append(toric.qubit_matrix)
                nr_failed_syndroms += 1


        success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, nr_failed_syndroms

def get_data(PATH):
    with open(PATH + '\MCTS_data_result.txt')as f:
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
    
    return P_success, P_error

def plot(system_size5, PATH5, plot_range):
    P_success5, P_error5 = get_data(PATH5)
    #P_success11, P_error11 = get_data(PATH11)
    #P_success13, P_error13 = get_data(PATH13)
     
    
    fig, ax = plt.subplots()

    ax.scatter(P_error5, P_success5,label='d = '+str(system_size5), color='steelblue', marker='o')
    #ax.scatter(P_error11, P_success11,label='d = '+str(system_size11), color='green', marker='^')
    #ax.scatter(P_error13, P_success13,label='d = '+str(system_size13), color='orange', marker='s')
    #ax.legend(fontsize = 14)
    ax.plot(P_error5,P_success5, color='steelblue')
    #ax.plot(P_error11,P_success11, color='green')
    #ax.plot(P_error13,P_success13, color='orange')
    ax.set_xlim(0.005,plot_range*0.01+0.005)
    plt.xlabel('$P_e$', fontsize=14)
    plt.ylabel('$P_s$', fontsize=14)
    plt.title('Prestanda för trändade agenter')
    plt.tick_params(axis='both', labelsize=14)
    #fig.set_figwidth(10)
    plt.savefig('Results'+'/Agent_Total_Result_Plot'+'.png')
    plt.show()

#Success = Error corrected and groundstate conserved
########################### Parameters ###########################                  ## Stats from 100 predictions ##
# d = 3, P_error = 0.1, disscount_backprop = 0.9, num_sim = 50              Success rate = 1.0, average number of steps = 2.5 
# d = 5, P_error = 0.1, disscount_backprop = 0.9, num_sim = 70              Success rate = 0.96, average number of steps = 5.6           
# d = 7, P_error = 0.1, disscount_backprop = 0.9, num_sim = 90
# d = 9, P_error = 0.1, disscount_backprop = 0.9, num_sim = 110
# d = 11, P_error = 0.1, disscount_backprop = 0.9, num_sim = 120


system_size = 5
num_sim = 100  #50
num_of_predictions = 1
plot_range = 20 # plot from P_error = 0.01 to plot_range*0.01
P_error = 0.2
PATH = 'Results/MCTS_prediction__20_04_19__22__51__26__'
#########################################################################################################################################
data_result = np.zeros((1, 2))
'''
timestamp = time.strftime("%y_%m_%d__%H__%M__%S__")
PATH = 'Results/MCTS_prediction__'+ timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)
'''


disscount_backprop = 0.8
cpuct = 0.9#np.sqrt(2) #OK
reward_multiplier = 100
device = 'cuda' #OK
grid_shift = system_size//2 #OK
prediction_list_p_error = [P_error] #OK
minimum_nbr_of_qubit_errors = 0 #int(system_size/2)+1
args = {'cpuct': cpuct, 'num_simulations':num_sim, 'grid_shift': system_size//2, 'discount_factor':disscount_backprop, \
    'reward_multiplier':reward_multiplier}

error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, nr_failed_syndroms = predictionMCTS(
    args = args,
    num_of_predictions=num_of_predictions, 
    num_of_steps=55, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=True,
    system_size = system_size)

win_rate = (num_of_predictions-(len(failed_syndroms)/2))/num_of_predictions
# save training settings in txt file 
data_result = np.append(data_result ,np.array([[prediction_list_p_error[0], win_rate]]), axis= 0)
np.savetxt(PATH + '/MCTS_data_result_0.005.txt', data_result, delimiter=',', fmt="%s")
print(error_corrected_list, 'error corrected')
print(ground_state_list, 'ground state conserved')
print(average_number_of_steps_list, 'average number of steps')
print('Nr of failed syndroms', nr_failed_syndroms, ' --> Win rate =',(num_of_predictions-nr_failed_syndroms)/num_of_predictions)

#plot(system_size, PATH, plot_range)


'''
P_error = 0.2
disscount_backprop = 0.9
cpuct = np.sqrt(2) #OK
reward_multiplier = 100
device = 'cuda' #OK
grid_shift = system_size//2 #OK
prediction_list_p_error = [P_error] #OK
minimum_nbr_of_qubit_errors = 0 #int(system_size/2)+1
args = {'cpuct': cpuct, 'num_simulations':num_sim, 'grid_shift': system_size//2, 'discount_factor':disscount_backprop, \
    'reward_multiplier':reward_multiplier}

error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, nr_failed_syndroms = predictionMCTS(
    args = args,
    num_of_predictions=num_of_predictions, 
    num_of_steps=50, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=False,
    system_size = system_size)

print(error_corrected_list, 'error corrected')
print(ground_state_list, 'ground state conserved')
print(average_number_of_steps_list, 'average number of steps')
print('Nr of failed syndroms', nr_failed_syndroms, ' --> Win rate =',(num_of_predictions-nr_failed_syndroms)/num_of_predictions)
'''
















