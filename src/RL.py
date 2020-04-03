# standard libraries
import numpy as np
import random
import time
from collections import namedtuple, Counter
import operator
import os
from copy import deepcopy
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
from .MCTS_vector import MCTS_vector


class RL():
    def __init__(self, Network, Network_name, system_size=int, p_error=0.1, replay_memory_capacity=int, learning_rate=0.00025,
                number_of_actions=3, max_nbr_actions_per_episode=50, device='cpu', replay_memory='uniform',
                cpuct=0.5, num_mcts_simulations=20, discount_factor=0.95, nr_trees=10):

        self.nr_trees = nr_trees
        # device
        self.device = device
        # Toric code
        if system_size%2 > 0:
            self.toric = [Toric_code(system_size) for _ in nr_trees]
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
        # hyperparameters MCTS
        self.tree_args = {'cpuct': cpuct, 'num_simulations':num_mcts_simulations, 'grid_shift': self.grid_shift, 'discount_factor': self.discount_factor}


    def save_network(self, PATH):
        torch.save(self.model, PATH)


    def load_network(self, PATH):
        self.model = torch.load(PATH, map_location='cpu')
        self.model = self.model.to(self.device)
    
    def experience_replay_vector(self, criterion, optimizer, batch_size):
        qval_perspective, weights, indices = self.memory.sample(batch_size, 0.4)

        batch_qvals, batch_perspectives = zip(*qval_perspective)
        batch_qvals = np.array(batch_qvals)

        batch_perspectives = np.array(batch_perspectives)
        batch_perspectives = np.convert_from_np_to_tensor(batch_perspectives).to(self.device)

        output = self.model.forward(batch_perspectives)

        loss = self.get_loss(criterion, optimizer, batch_qvals, output, weights, indices)

        loss.backward()
        optimizer.step()




    # def experience_replay(self, criterion, optimizer, batch_size):
    #     self.model.train()
    #     # get transitions and unpack them to minibatch
    #     transitions, weights, indices = self.memory.sample(batch_size, 0.4) # beta parameter 
    #     mini_batch = Transition(*zip(*transitions))
    #     # unpack action batch
    #     batch_actions = Action(*zip(*mini_batch.action))
    #     batch_actions = np.array(batch_actions.action) - 1
    #     batch_actions = torch.Tensor(batch_actions).long()
    #     batch_actions = batch_actions.to(self.device)
    #     # preprocess batch_input and batch_target_input for the network
    #     batch_state = self.get_batch_input(mini_batch.state)
    #     batch_next_state = self.get_batch_input(mini_batch.next_state)
    #     # preprocess batch_terminal and batch reward
    #     batch_terminal = convert_from_np_to_tensor(np.array(mini_batch.terminal)) 
    #     batch_terminal = batch_terminal.to(self.device)
    #     batch_reward = convert_from_np_to_tensor(np.array(mini_batch.reward))
    #     batch_reward = batch_reward.to(self.device)
    #     # compute policy net output
    #     output = self.model(batch_state)

    #     output2 = output.gather(1, batch_actions.view(-1, 1)).squeeze(1)    
    #     # compute target network output 
    #     mcts_output = self.get_mcts_output(batch_next_state, batch_size)
    #     y = batch_reward + (batch_terminal * self.discount_factor * mcts_output)
    #     # compute loss and update replay memory
    #     loss = self.get_loss(criterion, optimizer, y, output, weights, indices)
    #     # backpropagate loss
    #     loss.backward()
    #     optimizer.step()


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


    # def get_mcts_output(self, batch_next_state, batch_size):
    #     mcts = MCTS(self.model, self.device, self.tree_args, syndroms=batch_next_state.cpu().numpy())
    #     max_Q, _ = mcts.get_qs_actions()
    #     return max_Q.to(self.device)


    def get_batch_input(self, state_batch):
        batch_input = np.stack(state_batch, axis=0)
        batch_input = convert_from_np_to_tensor(batch_input)
        return batch_input.to(self.device)


    def train(self, epochs, training_steps=int, optimizer=str, batch_size=int, replay_start_size=int, reach_final_epsilon_cpuct=0.5,
              epsilon_start=1.0, num_of_epsilon_steps=10, epsilon_end=0.1,  cpuct_start=50, cpuct_end=0.1, num_of_epsilon_cpuct_steps=10):
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
            self.toric = [Toric_code(self.system_size) for _ in range(self.nr_trees)]
            # generate syndroms
            terminal_states = np.ones(self.nr_trees)
            for i in range(self.nr_trees):
                self.toric[i].generate_random_error(self.p_error)
                terminal_states[i] = self.toric[i].terminal_state(self.toric.current_state)

            mcts_vector = MCTS_vector(self.model, self.device, self.tree_args, toric_codes=self.toric)


            simulations = [50, 10]
            # solve one episode
            while not np.all(terminal_states == 0) and num_of_steps_per_episode < self.max_nbr_actions_per_episode and iteration < training_steps:

                start_time = time.time()
                
                simulation_index = num_of_steps_per_episode if num_of_steps_per_episode < len(simulations) else len(simulations)-1

                num_of_steps_per_episode += 1
                num_of_epsilon_cpuct_steps += 1
                steps_counter += 1
                iteration += 1

                mcts_vector.args['num_simulations']  = simulations[simulation_index]
                # select action using epsilon greedy policy
                
                Qvals, perspectives, actions = mcts_vector.get_Qvals()

                perspective_indecies, action_indecies = mcts_vector.get_best_indices(Qvals)
                best_actions = [actions[i][pi][ai] for i, pi, ai in zip(range(self.nr_trees), perspective_indecies, action_indecies)]               
                mcts_vector.next_step(best_actions)

                
                
                # save transition in memory
                #Alt 1: save all Qvals that we get:
                for i in range(self.nr_trees):
                    for Qs_list, perspective_list in zip(Qvals[i], perspectives[i]):
                        for Qs, perspective in zip(Qs_list, perspective_list):
                            self.memory.save(Qval_Perspective(deepcopy(Qs), deepcopy(perspective)), 10000)

                #Alt 2: save only the best:
                #for i, ai, pi in zip

                #Alt 3: Save only the ones that have been visited more than N times:

                # experience replay
                if steps_counter > replay_start_size:
                    update_counter += 1
                    self.experience_replay_vector(criterion,
                                            optimizer,
                                            batch_size) 

                # update epsilon and cpuct
                if (update_counter % epsilon_cpuct_update == 0):
                    epsilon = np.round(np.maximum(epsilon - epsilon_decay, epsilon_end), 3)
                    self.tree_args['cpuct'] = np.round(np.maximum(self.tree_args['cpuct'] - cpuct_decay, cpuct_end), 3)         

                # set next_state to new state 
                
                terminal_states = [tor.terminal_state(tor.current_state) for tor in self.toric]

                print('step time: {0:.3} s'.format(time.time() - start_time))

            

    def get_reward(self):
        terminal = np.all(self.toric.next_state==0)
        if terminal == True:
            reward = 100
        else:
            defects_state = np.sum(self.toric.current_state)
            defects_next_state = np.sum(self.toric.next_state)
            reward = defects_state - defects_next_state

        return reward


    def select_action(self, number_of_actions=3, epsilon=float):
        # set network in evluation mode 
        self.model.eval()
        # generate perspectives 
        perspectives = self.toric.generate_perspective(self.grid_shift, self.toric.current_state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            with torch.no_grad():
                policy_net_output = self.model(batch_perspectives)
                q_values_table = np.array(policy_net_output.cpu())
                row, col = np.where(q_values_table == np.max(q_values_table))
                perspective = row[0]
                max_q_action = col[0] + 1
                step = Action(batch_position_actions[perspective], max_q_action)
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            step = Action(batch_position_actions[random_perspective], random_action)  
        return step            


    def select_action_prediction(self):
        mcts = MCTS(deepcopy(self.model), self.device, self.tree_args, toric_codes=deepcopy(self.toric))
        _, action = mcts.get_qs_actions()
        return action


    def prediction(self, num_of_predictions=1, epsilon=0.0, num_of_steps=50, PATH=None, plot_one_episode=False, cpuct=0.0,
        show_network=False, show_plot=False, prediction_list_p_error=float, minimum_nbr_of_qubit_errors=0, print_Q_values=False, save_prediction=True):
        # load network for prediction and set eval mode 
        self.tree_args['cpuct'] = cpuct
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
                num_of_steps_per_episode = 0
                prev_action = 0
                terminal_state = 0
                # generate random syndrom
                self.toric = Toric_code(self.system_size)

                if minimum_nbr_of_qubit_errors == 0:
                    self.toric.generate_random_error(p_error)
                else:
                    self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
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

                    action = self.select_action_prediction()
                    
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

        for i in range(epochs):
            self.train(training_steps=training_steps,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    replay_start_size=replay_start_size,
                    epochs=epochs)
            print('training done, epoch: ', i+1)
            # evaluate network
            error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = self.prediction(num_of_predictions=num_of_predictions, epsilon=0.0, cpuct=0.0,
                                                                                                                                                                        prediction_list_p_error=prediction_list_p_error,                                                                                                                                                                        save_prediction=True,
                                                                                                                                                                        num_of_steps=num_of_steps_prediction)

            data_all = np.append(data_all, np.array([[self.system_size, self.network_name, i+1, self.replay_memory, self.device, self.learning_rate, optimizer,
            training_steps * (i+1), prediction_list_p_error[0], num_of_predictions, len(failed_syndroms)/2, error_corrected_list[0], ground_state_list[0], average_number_of_steps_list[0], self.p_error]]), axis=0)
            # save training settings in txt file 
            np.savetxt(directory_path + '/data_all.txt', data_all, 
                header='system_size, network_name, epoch, replay_memory, device, learning_rate, optimizer, total_training_steps, prediction_list_p_error, number_of_predictions, number_of_failed_syndroms, error_corrected_list, ground_state_list, average_number_of_steps_list, p_error_train', delimiter=',', fmt="%s")
            # save network
            step = (i + 1) * training_steps
            PATH = directory_path + '/network_epoch/size_{2}_{1}_epoch_{0}_memory_{5}_optimizer_{4}__steps_{3}_learning_rate_{6}.pt'.format(
                i+1, self.network_name, self.system_size, step, optimizer, self.replay_memory, self.learning_rate)
            self.save_network(PATH)
            
        return error_corrected_list

