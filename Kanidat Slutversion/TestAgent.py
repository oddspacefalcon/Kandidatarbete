import numpy as np
import time
import os
import torch
import copy
import _pickle as cPickle
from src.RL import RL
from src.TesterTree3 import TesterTree3
from src.TestTree2 import TestTree2
from src.TestTree import TestTree

from src.MCTS_Rollout import MCTS_Rollout
from src.MCTS_Rollout2 import MCTS_Rollout2

from src.toric_model import Toric_code
from src.toric_model import Action
from src.toric_model import Perspective

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.util import incremental_mean, convert_from_np_to_tensor, Transition, Action, Qval_Perspective


##########################################################################
def load_network(PATH):
    model = torch.load(PATH, map_location='cpu')
    model = model.to(device)
    return model

def step(action, syndrom):
    qubit_matrix = action.position[0]
    row = action.position[1]
    col = action.position[2]
    add_opperator = action.action
    rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)

    #if x or y
    if add_opperator == 1 or add_opperator ==2:
        if qubit_matrix == 0:
            syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[0][row][(col-1)%system_size] = (syndrom[0][row][(col-1)%system_size]+1)%2
        elif qubit_matrix == 1:
            syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
            syndrom[1][(row+1)%system_size][col] = (syndrom[1][(row+1)%system_size][col]+1)%2
    #if z or y
    if add_opperator == 3 or add_opperator ==2:
        if qubit_matrix == 0:
            syndrom[0][row][col] = (syndrom[0][row][col]+1)%2
            syndrom[0][(row-1)%system_size][col] = (syndrom[0][(row-1)%system_size][col]+1)%2
        elif qubit_matrix == 1:
            syndrom[1][row][col] = (syndrom[1][row][col]+1)%2
            syndrom[1][row][(col+1)%system_size] = (syndrom[1][row][(col+1)%system_size]+1)%2 

def rotate_state(state):
    vertex_matrix = state[0,:,:]
    plaquette_matrix = state[1,:,:]
    rot_plaquette_matrix = np.rot90(plaquette_matrix)
    rot_vertex_matrix = np.rot90(vertex_matrix)
    rot_vertex_matrix = np.roll(rot_vertex_matrix, 1, axis=0)
    rot_state = np.stack((rot_vertex_matrix, rot_plaquette_matrix), axis=0)
    return rot_state

def generate_perspective(grid_shift, state):
    def mod(index, shift):
        index = (index + shift) % system_size 
        return index
    perspectives = []
    vertex_matrix = state[0,:,:]
    plaquette_matrix = state[1,:,:]
    # qubit matrix 0
    for i in range(system_size):
        for j in range(system_size):
            if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
            plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                temp = Perspective(new_state, (0,i,j))
                perspectives.append(temp)
    # qubit matrix 1
    for i in range(system_size):
        for j in range(system_size):
            if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
            plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                new_state = np.roll(state, grid_shift-i, axis=1)
                new_state = np.roll(new_state, grid_shift-j, axis=2)
                new_state = rotate_state(new_state) # rotate perspective clock wise
                temp = Perspective(new_state, (1,i,j))
                perspectives.append(temp)
    
    return perspectives

def select_action_prediction(agent, state, grid_shift=int):
    # set network in eval mode
    agent.eval()
    # generate perspectives
    perspectives = generate_perspective(grid_shift, state)
    number_of_perspectives = len(perspectives)
    # preprocess batch of perspectives and actions 
    perspectives = Perspective(*zip(*perspectives))
    batch_perspectives = np.array(perspectives.perspective)
    batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
    batch_perspectives = batch_perspectives.to(device)
    batch_position_actions = perspectives.position
    # generate action value for different perspectives 
    with torch.no_grad():
        policy_net_output = agent(batch_perspectives)
        q_values_table = np.array(policy_net_output.cpu())
    row, col = np.where(q_values_table == np.max(q_values_table))
    perspective = row[0]
    max_q_action = col[0] + 1
    step = Action(batch_position_actions[perspective], max_q_action)

    return q_values_table, step

def get_possible_actions(state, grid_shift):
    perspectives = toric.generate_perspective(grid_shift, state)
    perspectives = Perspective(*zip(*perspectives))
    return [[Action(np.array(p_pos), x+1) for x in range(3)] for p_pos in perspectives.position]


def prediction(num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, cpuct=0.0,
    show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, print_Q_values=False, save_prediction=False,
    system_size=3):

    # load network for prediction and set eval mode
    if PATH != None:
        agent = load_network(PATH)
    agent.eval()
    grid_shift = system_size//2
    # init matrices 
    ground_state_list = np.zeros(len(prediction_list_p_error))
    error_corrected_list = np.zeros(len(prediction_list_p_error))
    average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
    mean_q_list = np.zeros(len(prediction_list_p_error))
    failed_syndroms = []
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

            # generate random syndrom
            toric = Toric_code(system_size)

            if minimum_nbr_of_qubit_errors == 0:
                toric.generate_random_error(p_error)
            else:
                toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
            terminal_state = toric.terminal_state(toric.current_state)
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                toric.plot_toric_code(toric.current_state, 'initial_syndrom')
            
            init_qubit_state = copy.deepcopy(toric.qubit_matrix)
            
            # solve syndrome
            while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1
                # choose greedy action

                _, action = select_action_prediction(agent, toric.current_state, grid_shift)
                prev_action = action
                toric.step(action)
                toric.current_state = toric.next_state
                terminal_state = toric.terminal_state(toric.current_state)
                if plot_one_episode == True and j == 0 and i == 0:
                   
                    toric.plot_toric_code(toric.current_state, 'step_'+str(num_of_steps_per_episode))

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

        success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error

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
            print('prediction nr', j)

            # generate random syndrom
            toric = Toric_code(system_size)

            if minimum_nbr_of_qubit_errors == 0:
                toric.generate_random_error(p_error)
            else:
                toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
            terminal_state = toric.terminal_state(toric.current_state)
            # plot one episode
            if plot_one_episode == True and j == 0 and i == 0:
                toric.plot_toric_code(toric.current_state, 'initial_syndrom')
            
            init_qubit_state = copy.deepcopy(toric.qubit_matrix)
            
            # solve syndrome
            while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                steps_counter += 1
                num_of_steps_per_episode += 1
                # choose MCTS action
             
                #mcts = TestTree('cpu', args, None, toric.current_state) #Double MCTS
                #_, action = mcts.get_Qvals()

                #mcts = TestTree('cpu', args, copy.deepcopy(toric), None) #random rollout MCTS
                #action = mcts.get_Qvals()

                mcts = MCTS_Rollout('cpu', args, copy.deepcopy(toric), None) #Classic rollout MCTS 
                action = mcts.get_qs_actions()
                
                print('___________________________________________________')
                print(toric.current_state)
                print('-----------')
                print('best action', action)
                add1 = np.sum(toric.current_state)

                prev_action = action
                toric.step(action)
                toric.current_state = toric.next_state
                terminal_state = toric.terminal_state(toric.current_state)

                print(toric.current_state)
                add2 = np.sum(toric.current_state)
                print('sum diff: ', add1-add2)
                print('Tot err left: ', add2)
                
                if plot_one_episode == True and j == 0 and i == 0:
                   
                    toric.plot_toric_code(toric.current_state, 'step_'+str(num_of_steps_per_episode))
                time.sleep(10)

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

        success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
        error_corrected_list[i] = success_rate
        ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
        ground_state_list[i] =  1 - ground_state_change
        average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)

    return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error

device = 'cuda'
num_of_predictions = 1
system_size = 5
grid_shift = system_size//2
prediction_list_p_error = [0.01]
minimum_nbr_of_qubit_errors = int(system_size/2)+1

P_error = 0.01

# Ok till sista d = 5
# d = 5, disscount_rollout = 0.75, num_sim_long = 100, rollout = 4, cpuct = 3, reward_multiplier = 3.5

disscount_rollout = 0.5 # 0.75
num_sim_long = 600  #50
rollout = 4
cpuct = 2
reward_multiplier = 1


args = {'cpuct': cpuct, 'num_simulations':num_sim_long, 'grid_shift': system_size//2, 'discount_factor':disscount_rollout, \
    'rollout_length':rollout, 'reward_multiplier':reward_multiplier}

error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = predictionMCTS(
    args = args,
    num_of_predictions=num_of_predictions, 
    num_of_steps=75, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=True,
    system_size = system_size)


print(error_corrected_list, 'error corrected')
print(ground_state_list, 'ground state conserved')
print(average_number_of_steps_list, 'average number of steps')
#print(failed_syndroms, 'failed syndroms')



'''
error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = prediction(
    num_of_predictions=num_of_predictions, 
    num_of_steps=75, 
    PATH=PATH, 
    prediction_list_p_error=prediction_list_p_error,
    minimum_nbr_of_qubit_errors=minimum_nbr_of_qubit_errors,
    plot_one_episode=True,
    system_size = system_size)
'''

'''
#............................Load network...........................
# Valid network names: NN_11, NN_17, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
network = NN_17
# This file is stored in the network folder and contains the trained agent.  
NETWORK_FILE_NAME = 'Size_5_NN_11'
# Path for the network to use for the prediction
PATH = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
agent = load_network(PATH)
'''

'''
#............................Generate errors...........................
# initial syndrome error generation 
# generate syndrome with error probability 0.1 
prediction_list_p_error = [0.1]
p_error = 0.1
# generate syndrome with a fixed amount of errors 
minimum_nbr_of_qubit_errors = 1 #int(system_size/2)+1 # minimum number of erorrs for logical qubit flip

toric = Toric_code(system_size)
if minimum_nbr_of_qubit_errors == 0:
    toric.generate_random_error(p_error)
else:
    toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
terminal_state = toric.terminal_state(toric.current_state)
#init_qubit_state = copy.deepcopy(toric.qubit_matrix)

def loopMCTS(state):
    counter = 0
    while True:
        counter += 1
        all_zeros = not np.any(state)
        if all_zeros:
            print('We Wooooon!!! ', state)
            win = 1
            break
        #............................Get best action from network...........................
        # Get Q-table
        q_values_table, best_action = select_action_prediction(agent, state, grid_shift)
        # Get action corresponding to max(Q)
        print('_____________________________')
        print('_____________________________')
        
        #print(q_values_table)
        current_state = copy.deepcopy(state)
        step(best_action, state)
        next_state = copy.deepcopy(state)
        print(current_state)
        print('-----------')
        print(best_action)
        print(next_state)
        add1 = np.sum(current_state)
        add2 = np.sum(next_state)
        print('sum diff: ', add1-add2)
        print('total errors left', add2)
        time.sleep(3)
    
    return

state = copy.deepcopy(toric.current_state)
loopMCTS(state)
'''

