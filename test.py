
from collections import Counter
import random
import traceback

import numpy as np
from CollaborativeAgent import CollaborativeAgent, transitionEvaluation
from UnstructuredAgent import UnstructuredLinearAgent, UnstructuredNNAgent
from environment import GridWorld, RandomStartGridWorld, majority_vote
from NaiveAgent import NaiveAgent
from plotter import animate_trajectory, plot_Q_functions, plot_episodes_optimality, plot_nn_directions, plot_nn_weights, plot_value_functions



class tester():

    def __init__(self, start_at_center = True, width = 10,height = 10,n_agents = 3,goal_reward = 100,empty_reward = -1,wall_reward = -10):

        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.goal_reward = goal_reward
        self.empty_reward = empty_reward
        self.wall_reward = wall_reward

        if start_at_center:
            self.env = GridWorld(self.width, self.height, self.n_agents, self.goal_reward, self.empty_reward, self.wall_reward)
        else:
            self.env = RandomStartGridWorld(self.width, self.height, self.n_agents, self.goal_reward, self.empty_reward, self.wall_reward)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Algorithm: Parametrized Value Iteration

    def runCollaborativeTest(self, evalEpisodes, collaboration = [0,1], plot_best_trajectory = False, plot_value_function = False, plot_optimality = False, save = None):

        states = [(x,y,z) for x in range(self.width) for y in range(self.height) for z in range(2**self.n_agents)] # z encode alive agents
        actions = ["↑", "↓", "←", "→"]
        gamma=0.95
        theta=1e-4
        delta = float('inf')
        valueFunction = [{s:self.empty_reward for s in states} for _ in range(self.n_agents)]
        k = [random.uniform(collaboration[0], collaboration[1]) for _ in range(self.n_agents)]

        # 1.Value function calculation

        while delta > theta: # Value iteration convergence for every agent
            delta = 0
            new_V = [{} for _ in range(self.n_agents)]

            for state in states:
                max_v = [float('-inf') for _ in range(self.n_agents)] 
                for action in actions:
                    newState, rewards = transitionEvaluation(state, action, self.env.goals.items(), self.height, self.width, self.goal_reward, self.empty_reward, self.wall_reward)
                    for i in range(self.n_agents):
                        # Notable attempt -> System does not converge with this configuration
                        # val = rewards[i] + gamma * (valueFunction[i][newState] + k[i] * sum([v[newState] for j,v in enumerate(valueFunction) if j!= i]))
                        val = rewards[i] + gamma * valueFunction[i][newState]
                        if max_v[i] < val:
                            max_v[i] = val
                for i in range(self.n_agents):
                    new_V[i][state] = max_v[i]
                    delta = max(delta, abs(valueFunction[i][state] - new_V[i][state]))
            valueFunction = new_V

        # 2.Value contamination

        Vcon = [{s:valueFunction[i][s] + k[i] * sum([v[s] for j,v in enumerate(valueFunction) if j!= i]) for s in states} for i in range(self.n_agents)]

        # 3.Runs

        agents = [CollaborativeAgent(i, self.env.getGoals(), self.height, self.width, Vcon[i], self.goal_reward, self.empty_reward, self.wall_reward) for i in range(self.n_agents)]
        all_episode_logs = []
        try:
            for episode in range(evalEpisodes):
                state = self.env.reset()
                log = {
                    "trajectory": [],
                    "rewards": {i: [] for i in range(self.n_agents)},
                    "votes": [],
                    "actions": [],
                    "active_agents": [],
                    "steps": 0,
                    "exited": False,
                    "optimal": 0
                }
                while any(self.env.is_active(agent.id) for agent in agents): # Repeat until at least an agent is alive
                    log["trajectory"].append(state)

                    votes = {}
                    for agent in agents:
                        if self.env.is_active(agent.id):
                            votes[agent.id]=agent.vote(state, self.env.agent_active) # Gather votes
                    action = majority_vote(votes.values()) # Select direction
                    alives = self.env.agent_active.copy()
                    next_state, rewards = self.env.step(action) # Move

                    if len(log["trajectory"]) > 1000:
                        log["exited"] = True
                        break

                    log["votes"].append(votes)
                    log["actions"].append(action)
                    log["active_agents"].append(alives)
                    for i in range(self.n_agents):
                        log["rewards"][i].append(rewards[i])

                    state = next_state
                if not log["exited"]:
                    log["steps"] = len(log["trajectory"])
                    log["optimal"] = log["steps"]/self.env.shortest_path_length(log["trajectory"][0])
                all_episode_logs.append(log)
        except Exception as err:
            animate_trajectory(log["trajectory"], log["active_agents"], (self.width, self.height), self.env.getGoals())
            raise err
        finally:
            exited = 0
            sumoptim = 0
            for l in all_episode_logs:
                if l["exited"]:
                    exited +=1
                else:
                    sumoptim += l['optimal']
            if evalEpisodes-exited !=0:
                avgOptim = sumoptim/(evalEpisodes-exited)
            else:
                avgOptim = None
            print(f"{save} Collaborative average optimality = {avgOptim}, err = {100*exited/evalEpisodes}")

            if plot_best_trajectory:
                best_episode = min(all_episode_logs, key=lambda d: d.get("optimal", 10))
                animate_trajectory(best_episode["trajectory"], best_episode["active_agents"], (self.width, self.height), self.env.getGoals())
            
            if plot_value_function:
                plot_value_functions(agents, self.width, self.height)

            if plot_optimality:
                plot_episodes_optimality([l.get('optimal', np.nan) for l in all_episode_logs], save=save)

        return avgOptim, 100*exited/evalEpisodes
        #return all_episode_logs

    # ----------------------------------------------------------------------------------------------------------------------------
    # Algorithm: Q-Learning

    def runNaiveTest(self, evalEpisodes, trainEpisodes = 1000, plot_best_trajectory = False, plot_Q_function = False, plot_optimality = False, save = None): 
        episodes = evalEpisodes + trainEpisodes
        agents = [NaiveAgent(i, self.env.getGoals(), self.height, self.width, q_init=self.empty_reward) for i in range(self.n_agents)]
        all_episode_logs = []
        try:
            for episode in range(episodes):
                state = self.env.reset()
                log = {
                    "trajectory": [],
                    "rewards": {i: [] for i in range(self.n_agents)},
                    "votes": [],
                    "actions": [],
                    "active_agents": [],
                    "steps": 0,
                    "exited": False,
                    "optimal": 0
                }
                while any(self.env.is_active(agent.id) for agent in agents): # Repeat until at least an agent is alive
                    log["trajectory"].append(state)
                    if episode == trainEpisodes:
                        for agent in agents:
                            agent.training = False
                    votes = {}
                    for agent in agents:
                        if self.env.is_active(agent.id):
                            votes[agent.id]=agent.vote(state, self.env.agent_active) # Gather votes
                    action = majority_vote(votes.values()) # Select direction
                    alives = self.env.agent_active.copy()
                    next_state, rewards = self.env.step(action) # Move
                    n_alives = self.env.agent_active

                    if episode < trainEpisodes:
                        for agent in agents:
                            if self.env.is_active(agent.id) or rewards[agent.id]==self.goal_reward: # To avoid dying before Q update
                                agent.update(state, alives,action,next_state, n_alives, rewards[agent.id])
                    else:
                        if len(log["trajectory"]) > 1000:
                            log["exited"] = True
                            break

                    log["votes"].append(votes)
                    log["actions"].append(action)
                    log["active_agents"].append(alives)
                    for i in range(self.n_agents):
                        log["rewards"][i].append(rewards[i])

                    state = next_state
                if not log["exited"]:
                    log["steps"] = len(log["trajectory"])
                    log["optimal"] = log["steps"]/self.env.shortest_path_length(log["trajectory"][0])
                all_episode_logs.append(log)
        except Exception as err:
            animate_trajectory(log["trajectory"], log["active_agents"], (self.width, self.height), self.env.getGoals())
            raise err
        
        finally:
            exited = 0
            sumoptim = 0
            for l in all_episode_logs[trainEpisodes:]:
                if l["exited"]:
                    exited +=1
                else:
                    sumoptim += l['optimal']
            if evalEpisodes-exited !=0:
                avgOptim = sumoptim/(evalEpisodes-exited)
            else:
                avgOptim = None

            print(f"{save} Naive average optimality = {avgOptim}, err = {100*exited/evalEpisodes}")

            if plot_best_trajectory:
                best_episode = min(all_episode_logs, key=lambda d: d.get("optimal", 10))
                animate_trajectory(best_episode["trajectory"], best_episode["active_agents"], (self.width, self.height), self.env.getGoals(), save_path="./travel.gif")
            
            if plot_Q_function:
                plot_Q_functions(agents, self.width, self.height)

            if plot_optimality:
                plot_episodes_optimality([l.get("optimal", np.nan) for l in all_episode_logs], training = trainEpisodes, save=save)
            
        return all_episode_logs
    
    # ----------------------------------------------------------------------------------------------------------------------------
    # Algorithm: SARSA with Neural Network Function Approximation

    def runUnstructuredTest(self, evalEpisodes, trainEpisodes = 1000, nnshape = [64,64], plot_best_trajectory = False, plot_nn = False, plot_optimality = False, save = None):
        episodes = evalEpisodes + trainEpisodes
        agents = [UnstructuredNNAgent(i, self.env.getGoals(), self.height, self.width, nnshape) for i in range(self.n_agents)]
        all_episode_logs = []
        try:
            for episode in range(episodes):
                state = self.env.reset()
                votes = {}
                for agent in agents:
                    if self.env.is_active(agent.id):
                        votes[agent.id]=agent.vote(state, self.env.agent_active, {}) # Gather votes
                action = majority_vote(votes.values()) # Select direction

                log = {
                    "trajectory": [state],
                    "rewards": {i: [] for i in range(self.n_agents)},
                    "votes": [votes],
                    "actions": [action],
                    "active_agents": [self.env.agent_active.copy()],
                    "steps": 0,
                    "exited": False,
                    "optimal": 0
                }
                
                while True: # Repeat until at least an agent is alive
                    alives = self.env.agent_active.copy()
                    next_state, rewards = self.env.step(action) # Move
                    next_alives = self.env.agent_active

                    if any(self.env.is_active(agent.id) for agent in agents):
                        prevVotes = votes
                        votes = {}
                        for agent in agents:
                            if self.env.is_active(agent.id):
                                votes[agent.id]=agent.vote(next_state, self.env.agent_active, prevVotes) # Gather votes
                        next_action = majority_vote(votes.values()) # Select direction

                        if episode < trainEpisodes:
                            for agent in agents:
                                if self.env.is_active(agent.id):
                                    agent.update(state, next_state, action, next_action, alives, next_alives, prevVotes, votes, rewards[agent.id])
                                if rewards[agent.id]==self.goal_reward:
                                    agent.lastUpdate(state, action, alives, prevVotes, self.goal_reward)
                        else:
                            if len(log["trajectory"]) > 1000:
                                log["exited"] = True
                                break
                    else: # After the last one standing
                        if episode < trainEpisodes:
                            for agent in agents:
                                if rewards[agent.id]==self.goal_reward:
                                    agent.lastUpdate(state, action, alives, prevVotes, self.goal_reward)
                        break

                    log["votes"].append(votes)
                    log["actions"].append(action)
                    log["active_agents"].append(alives)
                    for i in range(self.n_agents):
                        log["rewards"][i].append(rewards[i])

                    state = next_state
                    action = next_action
                    log["trajectory"].append(state)
                    if episode == trainEpisodes:
                        for agent in agents:
                            agent.training = False
                if not log["exited"]:
                    log["steps"] = len(log["trajectory"])
                    log["optimal"] = log["steps"]/self.env.shortest_path_length(log["trajectory"][0])
                    print(save, log["optimal"], episode, episodes)
                all_episode_logs.append(log)
        except Exception as err:
            animate_trajectory(log["trajectory"], log["active_agents"], (self.width, self.height), self.env.getGoals())
            traceback.print_exception(type(err), err, err.__traceback__)
            raise err

        finally:
            exited = 0
            sumoptim = 0
            for l in all_episode_logs[trainEpisodes:]:
                if l["exited"]:
                    exited +=1
                else:
                    sumoptim += l['optimal']
            if evalEpisodes-exited !=0:
                avgOptim = sumoptim/(evalEpisodes-exited)
            else:
                avgOptim = None

            print(f"{save} Unstructured average optimality = {avgOptim}, err = {100*exited/evalEpisodes}")

            if plot_best_trajectory:
                best_episode = min(all_episode_logs, key=lambda d: d.get("optimal", 10))
                animate_trajectory(best_episode["trajectory"], best_episode["active_agents"], (self.width, self.height), self.env.getGoals())
            
            if plot_nn:
                plot_nn_directions(agents, {agent.id:agent.model for agent in agents}, self.width, self.height)

            if plot_optimality:
                plot_episodes_optimality([l.get("optimal", np.nan) for l in all_episode_logs], training = trainEpisodes, save=save)
            
        #return all_episode_logs
        return avgOptim, 100*exited/evalEpisodes