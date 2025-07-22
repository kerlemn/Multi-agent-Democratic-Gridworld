import random
from environment import AbstractAgent

class NaiveAgent(AbstractAgent):
    def __init__(self, id, goals, height, width, q_init):
        super().__init__(id, goals, height, width)
        self.Q = {}
        self.q_init = q_init
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.1
        self.training = True

    def vote(self, state, alives):
        if random.random() < self.epsilon and self.training: # epsilon greedy on training time
            return random.choice(self.actions)

        x, y = state[0], state[1]
        z = self.encode(alives)

        max_q = max([self.Q.get((x, y, z, a), self.q_init) for a in self.actions])

        best_actions = [a for a in self.actions if self.Q.get((x, y, z, a), self.q_init) == max_q]

        return random.choice(best_actions)

    def update(self, state, alives, action,n_state, n_alives, reward):
        next_val = max(self.Q.get(n_state + (self.encode(n_alives),a,), self.q_init) for a in self.actions) # max[a]{ Q(S(t+1),a) }
        q = state + (self.encode(alives),action,)
        current_val = self.Q.get(q, self.q_init)
        self.Q[q] = current_val + self.alpha * (reward + self.gamma * next_val - current_val)
    

