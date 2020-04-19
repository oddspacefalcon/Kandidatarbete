import numpy as np
from .toric_model import Toric_code
from .util import Perspective, Action, convert_from_np_to_tensor
import math
import copy
import torch
import random
import sys
import time
EPS = 1e-8



class MCTS():

    def __init__(self, model, device, num_simulations, epsilon, discount_factor, grid_shift):


        class Node():
            def __init__(self):
                self.Q = None # torch.tensor(perspectives x actions) 
                self.perspectives = None # (batch_perspectives, batch_position_actions)
                self.child_nodes = {} # (str(state): Node)
                self.parent = None # Node
                self.visited_PQ = {} # (str(state), a): (perspective, Q)


        self.Node = Node
        self.device = device
        self.visited_temp = [] # (str(state), a, r)
        self.model = model
        self.num_simulations = num_simulations
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.grid_shift = grid_shift

    def backpropagate(self, node):
        if node.parent:
            for s, a, r in self.visited_temp[::-1]:
                Qa = r if r == 100 else self.discount_factor*node.Q.max().cpu().numpy() + r
                node = node.parent
                col = a.action - 1
                row = next((row for row, perspective in enumerate(node.perspectives[1]) if perspective == a.position), None)
                node.Q[row][col] = Qa
                node.visited_PQ[(s, a)] = (node.perspectives[0][row], Qa)

    def search(self, state, node, s, visited):
        # so no computation history is saved
        with torch.no_grad():
            if node.Q is None:
                if not np.all(state.current_state == 0):
                    # expand
                    perspectives = state.generate_perspective(self.grid_shift, state.current_state)
                    perspectives = Perspective(*zip(*perspectives))
                    batch_perspectives = np.array(perspectives.perspective)
                    batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
                    batch_perspectives = batch_perspectives.to(self.device)
                    batch_position_actions = perspectives.position
                    node.perspectives = (batch_perspectives, batch_position_actions)
                    node.Q = self.model(batch_perspectives)
                self.backpropagate(node)
                return
            else:
                # select new action using epsilon greedy
                rand = random.random()
                qvals = np.array(node.Q.cpu())
                if 1 - self.epsilon > rand:
                    action = None
                    while action is None:
                        row, col = np.where(qvals == np.max(qvals))
                        perspective_index = row[0]
                        action_index = col[0] + 1
                        a = Action(node.perspectives[1][perspective_index], action_index)
                        state.step(a)
                        s_n = np.array_str(state.next_state) 
                        if (s_n, a, self.get_reward(state)) not in self.visited_temp and (s_n, a) not in visited:
                            action = a
                        else:
                            qvals[row[0]][col[0]] = -1e6
                        state.step(a)
                else:
                    perspective_index = random.randint(0, qvals.shape[0] - 1)
                    action_index = random.randint(1, qvals.shape[1])
                    action = Action(node.perspectives[1][perspective_index], action_index)
                # take step
                state.step(action)
                self.visited_temp.append((s, action, self.get_reward(state)))
                state.current_state = state.next_state
                s1 = np.array_str(state.current_state)
                
                if s1 not in node.child_nodes:
                    # create new node
                    new_node = self.Node()
                    new_node.parent = node
                    node.child_nodes[s1] = new_node

                self.search(state, node.child_nodes[s1], s1, visited)

    def get_reward(self, state):
        if np.all(state.next_state == 0):
            return 100
        else:
            return np.sum(state.current_state) - np.sum(state.next_state)
                
    def get_tree(self, old_tree, state, visited):

        self.model.eval()

        if old_tree:
            simulations = self.num_simulations // 10
            root_node = old_tree
        else:
            simulations = self.num_simulations
            root_node = self.Node()

        for _ in range(simulations):
            self.search(copy.deepcopy(state), root_node, np.array_str(state.current_state), visited)
            self.visited_temp.clear()

        return root_node