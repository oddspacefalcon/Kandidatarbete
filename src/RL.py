# standard libraries
import numpy as np
import random
import time
from collections import namedtuple, Counter
import operator
import os
from copy import deepcopy
import copy
import heapq
# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
# import from other files
from .toric_model import Toric_code
from .toric_model import Action
from .toric_model import Perspective
from .Replay_memory import Replay_memory_uniform, Replay_memory_prioritized
# import networks 
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .util import incremental_mean, convert_from_np_to_tensor, Transition, Action, Qval_Perspective
from .MCTS import MCTS
from .MCTS_Rollout import MCTS_Rollout
from src.MCTS_Rollout2 import MCTS_Rollout2


class RL():
    def __init__(self, Network, Network_name, system_size=int, p_error=0.1, replay_memory_capacity=int, learning_rate=0.00025,
                number_of_actions=3, max_nbr_actions_per_episode=50, device='cpu', replay_memory='uniform',
                cpuct=0.5, num_mcts_simulations=20, discount_factor=0.95, nr_memories=5):

        # device
        self.device = device
        # Toric code
        if system_size%2 > 0:
            self.toric = Toric_code(system_size)
        else:
            raise ValueError('Invalid system_size, please use only odd system sizes.')
        self.grid_shift = int(system_size/2)
        self.max_nbr_actions_per_episode = max_nbr_actions_per_episode
        self.system_size = system_size
        self.p_error = p_error
        # Replay Memory
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_memory = replay_memory

        if self.replay_memory == 'proportional':
            self.memory = Replay_memory_prioritized(replay_memory_capacity, 0.6) # alpha
        elif self.replay_memory == 'uniform':
            self.memory = Replay_memory_uniform(replay_memory_capacity)
        else:
            raise ValueError('Invalid memory type, please use only proportional or uniform.')
        # Network
        self.network_name = Network_name
        self.network = Network
        if Network == ResNet18 or Network == ResNet34 or Network == ResNet50 or Network == ResNet101 or Network == ResNet152:
            self.model = self.network()
        else:
            self.model = self.network(system_size, number_of_actions, device)
        self.model = self.model.to(self.device)
        self.learning_rate = learning_rate
        # hyperparameters RL
        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor
        self.Nr_epoch = 0
        # hyperparameters MCTS
        
        
        ########################################## Set Parameters ##########################################            
        # d = 5, disscount_backprop = 0.9, num_sim = 70,  
        # d = 7, disscount_backprop = 0.9, num_sim = 90,            
        # d = 9, disscount_backprop = 0.9, num_sim = 110, 
        num_sim = 110  #Beror på system size

        disscount_backprop = 0.9 # OK
        cpuct = np.sqrt(2) #OK
        reward_multiplier = 100 #OK
        self.tree_args = {'cpuct': cpuct, 'num_simulations':num_sim, 'grid_shift': self.system_size//2, 'discount_factor':disscount_backprop, \
            'reward_multiplier':reward_multiplier}
        #####################################################################################################

    def save_network(self, PATH):
        torch.save(self.model, PATH)


    def load_network(self, PATH):
        self.model = torch.load(PATH, map_location='cpu')
        self.model = self.model.to(self.device)
    
    def experience_replay(self, criterion, optimizer, batch_size):
        qval_perspective, weights, indices = self.memory.sample(batch_size, 0.4)

        batch_qvals, batch_perspectives = zip(*qval_perspective)
        batch_qvals = np.array(batch_qvals)
        batch_qvals = convert_from_np_to_tensor(batch_qvals)
        batch_qvals = batch_qvals.to(self.device)

        batch_perspectives = np.array(batch_perspectives)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)

        output = self.model.forward(batch_perspectives)

        loss = self.get_loss(criterion, optimizer, batch_qvals, output, weights, indices)

        loss.backward()
        optimizer.step()

    def get_loss(self, criterion, optimizer, y, output, weights, indices):
        loss = criterion(y, output)
        optimizer.zero_grad()
        # for prioritized experience replay
        if self.replay_memory == 'proportional':
            loss = convert_from_np_to_tensor(np.array(weights)) * loss.cpu()
            priorities = loss
            priorities = np.absolute(priorities.detach().numpy())
            self.memory.priority_update(indices, priorities)
        return loss.mean()

    def get_batch_input(self, state_batch):
        batch_input = np.stack(state_batch, axis=0)
        batch_input = convert_from_np_to_tensor(batch_input)
        return batch_input.to(self.device)

    def train(self, epochs, training_steps=int, optimizer=str, batch_size=int, replay_start_size=int, reach_final_epsilon_cpuct=0.5,
              epsilon_start=1.0, num_of_epsilon_steps=10, epsilon_end=0.1,  cpuct_start=20, cpuct_end=0.1, num_of_epsilon_cpuct_steps=10):
        criterion = nn.MSELoss(reduction='none')
        # define criterion and optimizer
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        elif optimizer == 'Adam':    
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

        # init counters
        steps_counter = 0
        update_counter = 1
        iteration = 0

        # define epsilon and cpuct steps 
        epsilon = epsilon_start
        num_of_steps = np.round(training_steps/num_of_epsilon_steps)
        epsilon_decay = np.round((epsilon_start-epsilon_end)/num_of_epsilon_cpuct_steps, 5)
        cpuct_decay =  np.round((cpuct_start-cpuct_end)/num_of_epsilon_cpuct_steps, 5)
        epsilon_cpuct_update = num_of_steps * reach_final_epsilon_cpuct

        # main loop over training steps
        while iteration < training_steps:
            
            num_of_steps_per_episode = 0
            # initialize syndrom
            self.toric = Toric_code(self.system_size)
           
            # Generate random syndroms over interval P_error = [0.05, 0.15]
            #self.p_error = round(random.uniform(0.05,0.1), 2)
            
            # generate syndroms
            self.toric.generate_random_error(self.p_error)
            terminal_state = self.toric.terminal_state(self.toric.current_state)

            #start = time.time()
            #last_best_action = None
            #print('_____________________________________')
            #mcts = MCTS_Rollout('cpu', self.tree_args, copy.deepcopy(self.toric), None, last_best_action)
            mcts = MCTS_Rollout2('cpu', self.tree_args, copy.deepcopy(self.toric), None)
            #end = time.time()
            #print('Initiate MCTS:',end-start,' s')
            self.model.eval()

            #simulations = [100, 10]
            # solve one episode
            while terminal_state == 1 and num_of_steps_per_episode < self.max_nbr_actions_per_episode and iteration < training_steps:
                #print('-------------------------------------')
                num_of_steps_per_episode += 1
                num_of_epsilon_cpuct_steps += 1
                steps_counter += 1
                iteration += 1

                #mcts.args['num_simulations']  = simulations[simulation_index]
                # select action using epsilon greedy policy
                #start = time.time()
                Qvals, perspectives, actions, best_action = mcts.get_qs_actions()
                #end = time.time()
                #print('Run MCTS:',end-start,' s')
                #only put the perspectives that have been visited more than 1 time in the memory buffer
                Qvals, perspectives = mcts.get_memory_Qvals(Qvals, perspectives, actions, nr_min_visits=1)

                print('training steps:', iteration, 'Epoch nr:', self.Nr_epoch+1)

                mcts.next_step(best_action)
                # save transition in memory
                for Qs, perspective in zip(Qvals, perspectives):
                    self.memory.save(Qval_Perspective(deepcopy(Qs), deepcopy(perspective)), 10000)
  
                # experience replay
                if steps_counter > replay_start_size:
                    update_counter += 1
                    self.model.train()
                    self.experience_replay(criterion,
                                            optimizer,
                                            batch_size) 
                    self.model.eval()
 
                # update epsilon and cpuct
                if (update_counter % epsilon_cpuct_update == 0):
                    epsilon = np.round(np.maximum(epsilon - epsilon_decay, epsilon_end), 3)
                    self.tree_args['cpuct'] = np.round(np.maximum(self.tree_args['cpuct'] - cpuct_decay, cpuct_end), 3)         

                # set next_state to new state 
                #self.toric.step(best_action)
                #terminal_state = self.toric.terminal_state(self.toric.next_state)
                #self.toric.current_state = self.toric.next_state
                terminal_state = 0
                #print('set next_state to new state:',end-start,' s')
                
            
    def get_reward(self):
        terminal = np.all(self.toric.next_state==0)
        if terminal == True:
            reward = 100
        else:
            defects_state = np.sum(self.toric.current_state)
            defects_next_state = np.sum(self.toric.next_state)
            reward = defects_state - defects_next_state

        return reward

    def select_action_prediction(self, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
        # set network in eval mode
        self.model.eval()
        # generate perspectives
        perspectives = self.toric.generate_perspective(grid_shift, self.toric.current_state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        # generate action value for different perspectives 
        with torch.no_grad():
            policy_net_output = self.model(batch_perspectives)
            q_values_table = np.array(policy_net_output.cpu())
        
        #choose action using max(Q)
        row, col = np.where(q_values_table == np.max(q_values_table))
        perspective = row[0]
        max_q_action = col[0] + 1
        step = Action(batch_position_actions[perspective], max_q_action)
        q_value = q_values_table[row[0], col[0]]

        return step, q_value
    
    def prediction(self, num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, cpuct=0.0,
        show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, print_Q_values=False, save_prediction=True):
        # load network for prediction and set eval mode
        #self.tree_args['cpuct'] = cpuct
        if PATH != None:
            self.load_network(PATH)
        self.model.eval()
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
                print('prediction nr', j)
                num_of_steps_per_episode = 0
                prev_action = 0
                terminal_state = 0
                # generate random syndrom
                self.toric = Toric_code(self.system_size)
                #self.toric.generate_random_error(p_error)
                
                #self.toric.generate_random_error(p_error)
                ############################################
                
                if minimum_nbr_of_qubit_errors == 0:
                    self.toric.generate_random_error(p_error)
                else:
                    self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
                
                #############################################
                
                terminal_state = self.toric.terminal_state(self.toric.current_state)
                # plot one episode
                if plot_one_episode == True and j == 0 and i == 0:
                    self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')
                
                init_qubit_state = deepcopy(self.toric.qubit_matrix)
                
                # solve syndrome
                while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                    steps_counter += 1
                    num_of_steps_per_episode += 1
                    # choose greedy action
                    
                    action, q_value = self.select_action_prediction(number_of_actions=self.number_of_actions, 
                                                                    epsilon=epsilon,
                                                                    grid_shift=self.grid_shift,
                                                                    prev_action=prev_action)
                    #action = self.select_action_prediction()
                    prev_action = action
                    self.toric.step(action)
                    self.toric.current_state = self.toric.next_state
                    terminal_state = self.toric.terminal_state(self.toric.current_state)
                    if plot_one_episode == True and j == 0 and i == 0:
                        self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))

                # compute mean steps 
                mean_steps_per_p_error = incremental_mean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
                # save error corrected 
                error_corrected[j] = self.toric.terminal_state(self.toric.current_state) # 0: error corrected # 1: error not corrected    
                # update groundstate
                self.toric.eval_ground_state()                                                          
                ground_state[j] = self.toric.ground_state # False non trivial loops

                if terminal_state == 1 or self.toric.ground_state == False:
                    failed_syndroms.append(init_qubit_state)
                    failed_syndroms.append(self.toric.qubit_matrix)

            success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
            error_corrected_list[i] = success_rate
            ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
            ground_state_list[i] =  1 - ground_state_change
            average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)

        return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error

    def train_for_n_epochs(self, training_steps=int, epochs=int, num_of_predictions=100, num_of_steps_prediction=50, 
        optimizer=str, save=True, directory_path='network', prediction_list_p_error=[0.1],
        batch_size=32, replay_start_size=32):
        
        data_all = np.zeros((1, 15))
        data_result = np.zeros((1, 2))

        for i in range(epochs):
            self.train(training_steps=training_steps,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    replay_start_size=replay_start_size,
                    epochs=epochs)
            self.Nr_epoch = i+1
            print('training done, epoch: ', i+1)
            # evaluate network
            
            error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = self.prediction(num_of_predictions=num_of_predictions, epsilon=0.0, cpuct=0.0,
                                                                                                                                                                        prediction_list_p_error=prediction_list_p_error,                                                                                                                                                                        save_prediction=True,
                                                                                                                                                                        num_of_steps=num_of_steps_prediction)

            data_all = np.append(data_all, np.array([[self.system_size, self.network_name, i+1, self.replay_memory, self.device, self.learning_rate, optimizer,
            training_steps * (i+1), prediction_list_p_error[0], self.p_error, num_of_predictions, ground_state_list[0], average_number_of_steps_list[0], len(failed_syndroms)/2, error_corrected_list[0]]]), axis=0)
            # save training settings in txt file 
            np.savetxt(directory_path + '/data_all.txt', data_all, 
                header='system_size, network_name, epoch, replay_memory, device, learning_rate, optimizer, total_training_steps, prediction_list_p_error, p_error_train, number_of_predictions, ground_state_list, average_number_of_steps_list, number_of_failed_syndroms, error_corrected_list', delimiter=',', fmt="%s")
            
            data_result = np.append(data_result, np.array([[training_steps * (i+1), error_corrected_list[0]]]), axis=0)
            np.savetxt(directory_path + '/data_result.txt', data_result, 
                header='tot_training_steps, error_corrected', delimiter=',  ', fmt="%s")

            # save network
            step = (i + 1) * training_steps
            PATH = directory_path + '/network_epoch/size_{2}_{1}_epoch_{0}_memory_{5}_optimizer_{4}__steps_{3}_learning_rate_{6}.pt'.format(
                i+1, self.network_name, self.system_size, step, optimizer, self.replay_memory, self.learning_rate)
            self.save_network(PATH)
            
            
        return #error_corrected_list