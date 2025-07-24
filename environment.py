from itertools import permutations
import random
from collections import Counter

class GridWorld:
    def __init__(self, width, height, n_agents, reward, emptyReward, wallReward):
        self.width = width
        self.height = height
        self.reward = reward
        self.emptyReward = emptyReward
        self.wallReward = wallReward
        self.n_agents = n_agents
        self.startingPoint = (int(width/2),int(height/2))
        self.goals  = {}
        self.used_positions = {self.startingPoint}

        for i in range(n_agents): # Position goals randomly
            while True:
                goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
                if goal not in self.used_positions:
                    self.goals[i] = goal
                    self.used_positions.add(goal)
                    break
        self.reset()

    def is_active(self, agent_id):
        return self.agent_active[agent_id]
    
    def getGoals(self):
        return self.goals

    def step(self, action):
        rewards = {i: self.emptyReward for i in range(self.n_agents)}
        x, y = self.pos 
        if action == "↑":
            if y < self.height - 1:
                y += 1
            else:
                rewards = {i: self.wallReward for i in range(self.n_agents)}

        elif action == "↓":
            if y > 0:
                y -= 1
            else:
                rewards = {i: self.wallReward for i in range(self.n_agents)}

        elif action == "←":
            if x > 0:
                x -= 1
            else:
                rewards = {i: self.wallReward for i in range(self.n_agents)}

        elif action == "→":
            if x < self.width - 1:
                x += 1
            else:
                rewards = {i: self.wallReward for i in range(self.n_agents)}

        self.pos = (x, y)
        
        for i, goal in self.goals.items():
            if self.agent_active[i] and (x, y) == goal:
                rewards[i] = self.reward
                self.agent_active[i] = False

        return self.pos, rewards

    def reset(self):
        self.pos = self.startingPoint
        self.agent_active = [True for _ in self.goals]
        return self.pos
    

    def shortest_path_length(self, start): # From start to all the rewards
        min_path_len = float('inf')
        for order in permutations(self.goals.values()):
            current = start
            total_dist = 0
            for point in order:
                total_dist += manhattan(current, point)
                current = point
            min_path_len = min(min_path_len, total_dist)

        return int(min_path_len)

class RandomStartGridWorld(GridWorld):
    def reset(self): # Changes initial position at each reset
        self.used_positions.remove(self.startingPoint)
        while True:
            s = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if s not in self.used_positions:
                self.startingPoint = s
                self.used_positions.add(s)
                break
        self.pos = self.startingPoint
        self.agent_active = [True for _ in self.goals]
        return self.pos

class AbstractAgent:
    def __init__(self,id, goals, height, width):
        self.id = id
        self.goals = goals
        self.height = height
        self.width = width
        self.actions = ["↑", "↓", "←", "→"]

    def encode(self, bools):
        binary_str = ''.join(['1' if b else '0' for b in bools])
        return int(binary_str, 2)

def majority_vote(votes):
    counter = Counter(votes)
    max_count = max(counter.values())
    top_actions = [action for action, count in counter.items() if count == max_count]
    return random.choice(top_actions)

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])