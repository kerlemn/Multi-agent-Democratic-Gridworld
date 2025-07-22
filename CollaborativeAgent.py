import random
from environment import AbstractAgent

class CollaborativeAgent(AbstractAgent):
    def __init__(self, id, goals, height, width, V, goalReward, emptyReward, wallReward):
        super().__init__(id, goals, height, width)
        self.V = V
        self.gamma = 0.95
        self.goalReward = goalReward
        self.emptyReward = emptyReward
        self.wallReward = wallReward
     
    def vote(self, state, alives):
        policy=[]
        for action in self.actions: # Getting policy on state from value function
            n_state, rewards = transitionEvaluation(state+(self.encode(alives),), action, self.goals.items(), self.height, self.width, self.goalReward, self.emptyReward, self.wallReward)
            policy.append((sum(rewards.values()) + self.gamma * self.V[n_state], action))
        max_v = max([p[0] for p in policy])
        resAction = [p[1] for p in policy if p[0]==max_v]
        return random.choice(resAction)
    
def transitionEvaluation(state, action, goals, height, width, goalReward, emptyReward, wallReward):
    n_agents = len(goals)
    rewards = {i: emptyReward for i in range(n_agents)}
    x, y = state[0], state[1]
    if action == "↑":
        if y < height - 1:
            y += 1
        else:
            rewards = {i: wallReward for i in range(n_agents)}        

    elif action == "↓":
        if y > 0:
            y -= 1
        else:
            rewards = {i: wallReward for i in range(n_agents)}

    elif action == "←":
        if x > 0:
            x -= 1
        else:
            rewards = {i: wallReward for i in range(n_agents)}

    elif action == "→":
        if x < width - 1:
            x += 1
        else:
            rewards = {i: wallReward for i in range(n_agents)}

    z = state[2] 
    for i, goal in goals:
        bit_index = n_agents - 1 - i # Get the bit_index
        if (z >> bit_index) & 1 == 1 and (x, y) == goal: # Check if agent's bit_index is alive and this is its goal
            rewards[i] = goalReward
            z = z & ~(1 << bit_index) # Set agent's bit_index to dead
            
    assert 0 <= z < 2**n_agents, f"Invalid z: {z}"

    return (x,y,z), rewards