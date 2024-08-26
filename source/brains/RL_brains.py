import numpy as np
import time
import pickle
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from abc import ABC, abstractmethod
import os
from collections import defaultdict

plt.rcParams['figure.figsize'] = [20, 10]


class Agent(ABC):
    def __init__(self, actions_names, state_features, load_q_table=False):

        self.actions_list = actions_names  # string!
        self.state_features_list = state_features  # string!
        self.columns_q_table = actions_names + state_features  # string!

        self.q_table = None
        if load_q_table:
            if self.load_q_table():
                print("Load success")
            else:
                self.reset_q_table()
        else:
            self.reset_q_table()

        colours_list = ['green', 'red', 'blue', 'yellow', 'orange']
        self.action_to_color = dict(zip(self.actions_list, colours_list))
        self.size_of_largest_element = 800

        self.reference_list = []

    def reset_q_table(self):
        self.q_table = pd.DataFrame(columns=self.columns_q_table, dtype=np.float32)

        print("reset_q_table - self.q_table has shape = {}".format(self.q_table.shape))

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):

        self.check_state_exist(observation)

        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

        if not possible_actions:
            print("!!!!! WARNING - No possible_action !!!!!")

        if np.random.uniform() > greedy_epsilon:
            # choose best action

            # read the row corresponding to the state
            state_action = self.q_table.loc[
                (self.q_table[self.state_features_list[0]] == observation[0])
                & (self.q_table[self.state_features_list[1]] == observation[1])
                # & (self.q_table[self.state_features_list[2]] == observation[2])
                ]

            # only consider the action names - remove the state information
            state_action = state_action.filter(self.actions_list, axis=1)

            # shuffle - if different actions have equal q-values, chose randomly, not the first one
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            # restrict to allowed actions
            state_action = state_action.filter(items=possible_actions)
            # print("state_action 3/3 : %s" % state_action)

            # make decision
            if state_action.empty:
                action = random.choice(possible_actions)
                print('random action sampled among allowed actions')
            else:
                action = state_action.idxmax(axis=1)
                # Return index of first occurrence of maximum over requested axis (with shuffle)

            # get first element of the pandas series
            action_to_do = action.iloc[0]
            # print("\tBEST action = %s " % action_to_do)

        else:
            # choose random action
            action_to_do = np.random.choice(possible_actions)
            # print("\t-- RANDOM action= %s " % action_to_do)

        return action_to_do

    def compare_reference_value(self):

        state = [16, 3]
        action_id = 0  # "no change"
        self.check_state_exist(state)
        id_row_previous_state = self.get_id_row_state(state)
        res = self.q_table.loc[id_row_previous_state, self.actions_list[action_id]]
        # should be +40
        # print("reference_value = {}".format(res))
        self.reference_list.append(res)
        return res

    @abstractmethod
    def learn(self, *args):

        pass

    def check_state_exist(self, state):
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            new_data = np.concatenate((np.array(len(self.actions_list) * [0]), np.array(state)), axis=0)
            new_row = pd.Series(new_data, index=self.q_table.columns)
            self.q_table = self.q_table.append(new_row, ignore_index=True)

    def get_id_row_state(self, s):

        id_list_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == s[0]) &
                                           (self.q_table[self.state_features_list[1]] == s[1])].tolist()
        id_row_state = id_list_state[0]
        return id_row_state

    def load_q_table(self, weight_file=None):

        try:
            # from pickle
            if weight_file is None:
                grand_grand_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                results_dir = os.path.abspath(grand_grand_parent_dir + "/results/simple_road/" + "q_table" + '.pkl')
                self.q_table = pd.read_pickle(results_dir)
            else:
                self.q_table = pd.read_pickle(weight_file)
            return True

        except Exception as e:
            print(e)
        return False

    def save_q_table(self, save_directory):

        filename = "q_table"
        # sort series according to the position
        self.q_table = self.q_table.sort_values(by=[self.state_features_list[0]])
        try:
            # to pickle
            self.q_table.to_pickle(save_directory + filename + ".pkl")
            print("Saved as " + filename + ".pkl")

        except Exception as e:
            print(e)

    def print_q_table(self):

        self.q_table = self.q_table.sort_values(by=[self.state_features_list[0]])
        # print(self.q_table.head())
        print(self.q_table.to_string())

    def plot_q_table(self, folder, display_flag):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # not to overlap scatters
        shift = 0.2

        min_value = min(self.q_table[self.actions_list].min(axis=0))
        max_value = max(self.q_table[self.actions_list].max(axis=0))

        scale_factor = self.size_of_largest_element / max(max_value, abs(min_value))

        i = 0
        for action in self.actions_list:
            colour_for_action = self.action_to_color[action]
            colour_for_action_neg = colour_for_action
            markers = ['P' if i > 0 else 's' for i in self.q_table[action]]
            sizes = [scale_factor * (abs(i)) for i in self.q_table[action]]
            colours = [colour_for_action if i > 0 else colour_for_action_neg for i in self.q_table[action]]
            for x, y, m, s, c in zip(self.q_table[self.state_features_list[0]],
                                     self.q_table[self.state_features_list[1]], markers, sizes, colours):
                ax1.scatter(x, y + i * shift, alpha=0.8, c=c, marker=m, s=s)
            i += 1

        # custom labels
        labels_list = []
        for action in self.actions_list:
            label = patches.Patch(color=self.action_to_color[action], label=action)
            labels_list.append(label)
        plt.legend(handles=labels_list)

        # plot decoration
        plt.title('Normalized Q(s,a) - distinguishing positive and negative values with marker type')
        plt.xlabel(self.state_features_list[0])
        plt.ylabel(self.state_features_list[1])
        plt.xticks(np.arange(min(self.q_table[self.state_features_list[0]]),
                             max(self.q_table[self.state_features_list[0]]) + 1, 1.0))
        plt.grid(True, alpha=0.2)
        ax1.set_facecolor('silver')
        plt.savefig(folder + "plot_q_table.png", dpi=800)
        if display_flag:
            plt.show()

    def plot_optimal_actions_at_each_position(self, folder, display_flag):
        # scaling
        min_value = min(self.q_table[self.actions_list].min(axis=0))
        max_value = max(self.q_table[self.actions_list].max(axis=0))
        scale_factor = self.size_of_largest_element / max(max_value, abs(min_value))

        # look for the best action for each state
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        for index, row in self.q_table.iterrows():
            action_value = row.filter(self.actions_list, axis=0)
            action = action_value.idxmax()
            value = action_value.max()
            x = row[self.state_features_list[0]]
            y = row[self.state_features_list[1]]
            c = self.action_to_color[action]

            if value > 0:
                m = 'P'
            else:
                m = 's'
            s = scale_factor * abs(value)
            ax2.scatter(x, y, alpha=0.8, c=c, marker=m, s=s)

        # custom labels
        labels_list = []
        for action in self.actions_list:
            label = patches.Patch(color=self.action_to_color[action], label=action)
            labels_list.append(label)
        plt.legend(handles=labels_list)

        # plot decoration
        plt.title('Normalized max[Q(s,a)][over a] - Optimal actions - randomly selected if equal')
        plt.xlabel(self.state_features_list[0])
        plt.ylabel(self.state_features_list[1])
        plt.xticks(np.arange(min(self.q_table[self.state_features_list[0]]),
                             max(self.q_table[self.state_features_list[0]]) + 1, 1.0))
        plt.grid(True, alpha=0.2)
        ax2.set_facecolor('silver')
        plt.savefig(folder + "plot_optimal_actions_at_each_position.png", dpi=800)
        if display_flag:
            plt.show()


class SarsaTable(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(SarsaTable, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, a_, termination_flag, gamma, learning_rate):

        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the pair (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
        else:
            # next state is not terminal
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + gamma * q_expected

        # update q-value - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


# to compute the q_predict, make the average of q-values based on probabilities of each action
class ExpectedSarsa(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(ExpectedSarsa, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, termination_flag, greedy_epsilon, gamma, learning_rate):

        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal - goal state has no value
            q_target = r
            # Trying to reduce chance of random action as we train the model

        else:
            # next state is not terminal
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)

            q_max = max(filtered_row)
            # print("q_max = \n{}".format(q_max))

            q_mean = 0
            if len(filtered_row):
                q_mean = sum(filtered_row) / len(filtered_row)
            # print("q_mean = \n{}".format(q_mean))

            q_expected = (1 - greedy_epsilon) * q_max + greedy_epsilon * q_mean
            # print("q_expected = \n{}".format(q_expected))

            q_target = r + gamma * q_expected

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


# off-policy. Q-learning = sarsa_max
class QLearningTable(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(QLearningTable, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, termination_flag, gamma, learning_rate):

        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
            # Trying to reduce chance of random action as we train the model.

        else:
            # next state is not terminal
            # consider the best value of the next state. Q-learning = sarsa_max
            # using max to evaluate Q(s_, a_) - Q-learning is therefore said "off-policy"
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)
            # print(s)
            # print("filtered_row = \n{}".format(filtered_row))
            # print("max(filtered_row) = \n{}".format(max(filtered_row)))
            q_expected = max(filtered_row)
            q_target = r + gamma * q_expected
            # q_target = r + gamma * self.q_table.loc[id_row_next_state, :].max()

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


class SarsaLambdaTable(Agent):
    def __init__(self, actions, state, load_q_table=False,
                 trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, state, load_q_table)
        self.lambda_trace_decay = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def reset_eligibility_trace(self):
        self.eligibility_trace[self.actions_list] = 0.0


    def check_state_exist(self, state):
        # try to find the index of the state - same as for the parent Class
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            # append new state to q table: Q(a,s)=0 for each action a
            new_data = np.concatenate((np.array(len(self.actions_list) * [0]), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=self.q_table.columns)

            # add new row in q_table
            self.q_table = self.q_table.append(new_row, ignore_index=True)

            # also add it to the eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(new_row, ignore_index=True)

    def learn(self, s, a, r, s_, a_, termination_flag, gamma, learning_rate):

        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            q_target = r
        else:
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + gamma * q_expected

        # TD-error
        error = q_target - q_predict
        self.eligibility_trace.loc[id_row_previous_state, a] = 1
        self.q_table[self.actions_list] += learning_rate * error * self.eligibility_trace[self.actions_list]
        self.eligibility_trace[self.actions_list] *= gamma * self.lambda_trace_decay


# Monte Carlo Control
class MC(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(MC, self).__init__(actions, state, load_q_table)
        self.nA = len(actions)
        # self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def compare_reference_value(self):
        state = (16, 3)
        action_id = 0  # "no change"
        res = self.q_table[state][action_id]
        # should be +40
        print("reference_value = {}".format(res))
        return res

    def reset_q_table(self):
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):
        observation = tuple(observation)

        # apply action masking
        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

        # Epsilon-greedy action selection
        if np.random.uniform() > greedy_epsilon:
            # choose best action

            state_action = copy(self.q_table[observation])


            # restrict to allowed actions
            for action in self.actions_list:
                if action not in possible_actions:
                    action_id = self.actions_list.index(action)
                    state_action[action_id] = -np.inf  # using a copy

            # make decision
            if np.all(np.isneginf([state_action])):
                action_id = random.choice(possible_actions)
                print('random action sampled among allowed actions')
            else:
                action_id = np.argmax(state_action)
                # Return index of first occurrence of maximum over requested axis (with shuffle)
            action_to_do = self.actions_list[action_id]
        else:
            action_to_do = np.random.choice(possible_actions)

        return action_to_do

    def learn(self, episode, gamma, learning_rate):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            action_id = self.actions_list.index(actions[i])
            old_q = self.q_table[state][action_id]
            self.q_table[state][action_id] = old_q + learning_rate * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_q)

    def save_q_table(self, save_directory):
        filename = "q_table"
        try:
            # to pickle
            output = open(save_directory + filename + ".pkl", 'wb')

            pickle.dump(dict(self.q_table), output)
            output.close()
            print("Saved as " + filename + ".pkl")

        except Exception as e:
            print(e)

    def print_q_table(self):
        # sort series according to the position
        q_table_dict = dict(self.q_table)
        q_table_pandas = pd.DataFrame(columns=self.columns_q_table, dtype=np.float32)
        for state, q_values in q_table_dict.items():
            new_data = np.concatenate((np.array(q_values), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=q_table_pandas.columns)
            q_table_pandas = q_table_pandas.append(new_row, ignore_index=True)

        q_table_pandas = q_table_pandas.sort_values(by=[self.state_features_list[0]])
        print(q_table_pandas.to_string())

    def load_q_table(self, weight_file=None):
        try:
            # from pickle
            print(weight_file)
            loaded_dict = pd.read_pickle(weight_file)
            print(type(loaded_dict))

            self.q_table = defaultdict(lambda: np.zeros(self.nA))

            for state, value in loaded_dict.items():
                for i, q in enumerate(value):
                    # print("state = {}".format(state))
                    # print("i = {}".format(i))
                    # print("q = {}".format(q))
                    self.q_table[state][i] = q
            return True

        except Exception as e:
            print(e)
        return False


# Model-based
class DP(Agent):
    def __init__(self, actions, state, env, gamma, load_q_table=False):
        super(DP, self).__init__(actions, state, load_q_table)
        self.env = env
        self.nA = len(actions)
        self.n_position = 20
        self.n_velocity = 6
        self.gamma = gamma

    def learn(self):
        pass

    def get_value_from_state(self, state, action):
        """
        debug: make one step in the environment
        """
        [p, v] = state
        self.env.reset()
        self.env.move_to_state([p, v])  # teleportation
        next_observation, reward, termination_flag, _ = self.env.step(action)
        return next_observation, reward, termination_flag

    def run_policy(self, policy, initial_state, max_nb_steps=100):
        self.env.reset()

        # from Policy to Value Functions - for debug
        v_table = self.policy_evaluation(policy=policy)
        q_table = self.q_from_v(v_table)

        current_observation = initial_state
        self.env.move_to_state(initial_state)  # say the env to move to state [p][v]
        return_of_episode = 0
        trajectory = []

        step_count = 0
        while step_count < max_nb_steps:
            step_count += 1

            policy_for_this_state = policy[current_observation[0], current_observation[1]]
            print("policy_for_this_state = {}".format(policy_for_this_state))
            print("q_values_for_this_state = {}".format(q_table[current_observation[0], current_observation[1]]))

            action_id = np.argmax(policy[current_observation[0], current_observation[1]])
            action = self.actions_list[action_id]
            print("action = {}".format(action))

            trajectory.append(current_observation)
            trajectory.append(action)

            next_observation, reward, termination_flag, _ = self.env.step(action)
            print(" {}, {}, {} = results".format(next_observation, reward, termination_flag))

            return_of_episode += reward

            current_observation = next_observation
            if termination_flag:
                trajectory.append(next_observation)
                break

        print("return_of_episode = {}".format(return_of_episode))
        print("Trajectory = {}".format(trajectory))
        return return_of_episode, trajectory

    def q_from_v(self, v_table):
        q_table = np.ones((self.n_position, self.n_velocity, self.nA))
        for p in range(self.n_position):
            for v in range(self.n_velocity):
                masked_actions_list = self.env.masking_function([p, v])
                possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

                for action_id in range(self.nA):
                    self.env.move_to_state([p, v])  # say the env to move on on state [p][v]
                    action = self.actions_list[action_id]
                    if action in possible_actions:
                        next_observation, reward, termination_flag, _ = self.env.step(action)
                        prob = 1  # it is a deterministic environment
                        if termination_flag:
                            q_table[p][v][action_id] = prob * reward

                        else:
                            next_p = next_observation[0]
                            next_v = next_observation[1]
                            q_table[p][v][action_id] = prob * (reward + self.gamma * v_table[next_p][next_v])

                    else:
                        q_table[p][v][action_id] = -np.inf  # masked action
        return q_table

    def policy_improvement(self, v_table):
        policy = np.zeros([self.n_position, self.n_velocity, self.nA]) / self.nA
        for p in range(self.n_position):
            for v in range(self.n_velocity):
                q_table = self.q_from_v(v_table)
                best_a = np.argwhere(q_table[p][v] == np.max(q_table[p][v])).flatten()
                policy[p][v] = np.sum([np.eye(self.nA)[i] for i in best_a], axis=0) / len(best_a)

        return policy

    # truncated policy_evaluation
    def policy_evaluation(self, theta_value_function=10e-3, policy=None, max_counter=1e3):
        if policy is None:
            policy = np.ones([self.n_position, self.n_velocity, self.nA]) / self.nA  # random_policy
        # initialize arbitrarily
        v_table = np.zeros((self.n_position, self.n_velocity))
        counter = 0
        while counter < max_counter:

            counter += 1
            if counter % 1000 == 0:
                print(" --- {} policy_evaluation --- ".format(counter))
            delta_value_functions = 0

            # loop over all possible states (p, v)
            for p in range(self.n_position):
                for v in range(self.n_velocity):

                    v_state = 0
                    masked_actions_list = self.env.masking_function([p, v])

                    possible_actions = [action for action in self.actions_list if action not in masked_actions_list]
                    for action_id, action_prob in enumerate(policy[p][v]):
                        self.env.move_to_state([p, v])  # say the env to move on on state [p][v]

                        action = self.actions_list[action_id]
                        if action in possible_actions:
                            next_observation, reward, termination_flag, _ = self.env.step(action)
                            prob = 1  # deterministic environment
                            next_p = next_observation[0]
                            next_v = next_observation[1]


                            if termination_flag:
                                v_state += action_prob * prob * reward
                            else:
                                v_state += action_prob * prob * (reward + self.gamma * v_table[next_p][next_v])
                    delta_value_functions = max(delta_value_functions, np.abs(v_table[p][v] - v_state))
                    v_table[p][v] = v_state

            if delta_value_functions < theta_value_function:
                break
        return v_table

    # truncated Policy_Iteration
    def policy_iteration(self, theta_value_function=1e-3, theta_final_value_function=1e-5, max_counter=1e3):
        time_start = time.time()

        policy = np.zeros([self.n_position, self.n_velocity, self.nA]) / self.nA
        counter = 0
        v_table = None
        delta_policy = None

        while counter < max_counter:
            counter += 1
            intermediate_time = time.time()
            duration = intermediate_time - time_start
            print(" - {}-th iteration in Policy_Iteration - duration = {:.2f} - delta_policy = {}".format(
                counter, duration, delta_policy))

            # 1- Evaluation: For fixed current policy, find values with policy evaluation
            v_table = self.policy_evaluation(theta_value_function=theta_value_function,
                                             policy=policy,
                                             max_counter=max_counter)
            new_policy = self.policy_improvement(v_table)

            # 2- Improvement : For fixed values, get a better policy using policy extraction (One-step look-ahead)
            # OPTION 1: stop if the policy is unchanged after an improvement step

            # OPTION 2: stop if the value function estimates for successive policies has converged
            # i.e. if policies have similar value functions
            delta_policy = np.max(abs(self.policy_evaluation(policy=policy,
                                                             theta_value_function=theta_value_function,
                                                             max_counter=max_counter)
                                      - self.policy_evaluation(policy=new_policy,
                                                               theta_value_function=theta_value_function,
                                                               max_counter=max_counter)
                                      ))
            if delta_policy < theta_final_value_function:
                break

            policy = copy(new_policy)

        if counter == max_counter:
            print("Policy_Iteration() stops because of max_counter = {}".format(max_counter))
        else:
            print("Policy_Iteration() stops because of theta_value_function = {}".format(theta_value_function))

        time_stop = time.time()
        duration = time_stop - time_start
        print("Duration of Policy Iteration = {:.2f} - counter = {} - delta_policy = {}".format(duration, counter,
                                                                                                delta_policy))

        return policy, v_table

    def value_iteration(self, theta_value_function=1e-5, max_counter=1e3):
        time_start = time.time()

        # initialize V arbitrarily
        v_table = np.zeros((self.n_position, self.n_velocity))
        counter = 0
        delta_value_functions = None

        while counter < max_counter:
            counter += 1
            intermediate_time = time.time()
            duration = intermediate_time - time_start
            print(" - {}-th iteration in Value_Iteration - duration = {:.2f} - delta_value_functions = {}".format(
                counter, duration, delta_value_functions))

            delta_value_functions = 0
            # loop over all states
            for p in range(self.n_position):
                for v in range(self.n_velocity):
                    value = v_table[p][v]

                    # usually policy evaluation to update v_table[state] with Bellman. Here, one sweep only
                    q_table = self.q_from_v(v_table)
                    v_table[p][v] = np.max(q_table[p][v])

                    # check how much the value has changed
                    delta_value_functions = max(delta_value_functions, abs(v_table[p][v] - value))
            if delta_value_functions < theta_value_function:
                break
        # at this point, we have the Optimal Value_Function
        # let's obtain the corresponding policy
        policy = self.policy_improvement(v_table)

        if counter == max_counter:
            print("Value_Iteration() stops because of max_counter = {}".format(max_counter))
        else:
            print("Value_Iteration() stops because of theta_value_function = {}".format(theta_value_function))

        time_stop = time.time()
        duration = time_stop - time_start
        print("Duration of Value Iteration = {:.2f} - counter = {} - delta_value_functions = {}".format(
            duration, counter, delta_value_functions))

        return policy, v_table
