import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers
from collections import Counter
import random
from environment import AbstractAgent


class UnstructuredLinearAgent(AbstractAgent): # Notable attempt but gets stuck too often
    def __init__(self, id, goals, height, width):
        super().__init__(id, goals, height, width)
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.01
        self.training = True
        self.weights = np.zeros(self.height + self.width + 7 + len(goals)) # one-hot(X) + one-hot(Y) + one-hot(other-alives) + previousVotes(actions) + one-hot(actions) -> self.height + self.width + n_agents -1 + 4 + 4

    def _featurize(self, state, alive, votes, action):
        x, y = state
        count = Counter(votes)
        features = \
            [int(j == x) for j in range(self.width)] + \
            [int(j == y) for j in range(self.height)] + \
            [1 if b else 0 for i,b in enumerate(alive) if i != self.id] + \
            [count[a]/4 for a in self.actions] + \
            [int(a == action) for a in self.actions]
        return np.array(features, dtype=np.float32)

    def vote(self, state, alive, votes):
        if random.random() < self.epsilon and self.training: # epsilon greedy on training time
            return random.choice(self.actions)

        qs = {}

        for action in self.actions:
            feat = self._featurize(state, alive, votes, action)
            qs[action] = self.weights @ feat 

        max_q = max(qs.values())

        best_actions = [a for a in self.actions if qs[a] == max_q]

        return random.choice(best_actions)
    
    def update(self, state, state_, action, action_, alive, alive_, votes, votes_, reward):

        feat = self._featurize(state, alive, votes, action)
        q = self.weights @ feat

        feat_ = self._featurize(state_, alive_, votes_, action_)
        q_ = self.weights @ feat_

        td_err = reward + self.gamma * q_ - q

        self.weights = self.weights + self.alpha * td_err * feat

    def lastUpdate(self, state, action, alive, votes, reward):

        feat = self._featurize(state, alive, votes, action)
        q = self.weights @ feat

        td_err = reward - q

        self.weights = self.weights + self.alpha * td_err * feat

featurize_cache = {}

class UnstructuredNNAgent(AbstractAgent):
    def __init__(self, id, goals, height, width, nnshape):
        super().__init__(id, goals, height, width)
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.0001
        self.training = True
        self.input_dim = self.height + self.width + 3 + len(goals) # one-hot(X) + one-hot(Y) + one-hot(other-alives) + previousVotes(actions) -> self.height + self.width + n_agents -1 + 4
        self.output_dim = 4 # len(actions)
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.model = self._build_model(nnshape)
        self.optimizer = optimizers.SGD(learning_rate=self.alpha)
    
    def _build_model(self, nnshape):
        lay = [layers.Dense(l, activation='relu') for l in nnshape]
        model = models.Sequential(
            [layers.Input(shape=(self.input_dim,))] + lay + [layers.Dense(self.output_dim)]
        )
        return model
    
    def _featurize(self, state, alive, votes):
        key = (tuple(state), tuple(alive), tuple(votes), self.id)
        if key in featurize_cache:
            return featurize_cache[key]
        x, y = state
        count = Counter(votes)
        features = \
            [int(j == x) for j in range(self.width)] + \
            [int(j == y) for j in range(self.height)] + \
            [1 if b else 0 for i,b in enumerate(alive) if i != self.id] + \
            [count[a]/4 for a in self.actions]
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        featurize_cache[key] = features
        return features

    def vote(self, state, alive, votes):
        if random.random() < self.epsilon and self.training:
            return random.choice(self.actions)

        feat = self._featurize(state, alive, votes)
        q_values = self.model(feat, training=False).numpy()[0]
        max_q = np.max(q_values)

        best_actions = [a for a in self.actions if q_values[self.action_to_index[a]] == max_q]
        return random.choice(best_actions)

    def update(self, state, state_, action, action_, alive, alive_, votes, votes_, reward):
        feat = self._featurize(state, alive, votes)
        feat_ = self._featurize(state_, alive_, votes_)

        q_next = self.model(feat_, training=False)[0, self.action_to_index[action_]]
        target = reward + self.gamma * q_next
        action_idx = tf.convert_to_tensor(self.action_to_index[action], dtype=tf.int32)

        apply_gradient(self.model,feat, target, action_idx, self.optimizer)

    def lastUpdate(self, state, action, alive, votes, reward):
        feat = self._featurize(state, alive, votes)
        target = tf.convert_to_tensor(reward, dtype=tf.float32)
        action_idx = tf.convert_to_tensor(self.action_to_index[action], dtype=tf.int32)
        apply_gradient(self.model,feat, target, action_idx, self.optimizer)

@tf.function
def apply_gradient(model, feat, target, action_idx, optimizer):
    with tf.GradientTape() as tape:
        q_values = model(feat, training=True)
        q = q_values[0, action_idx]
        loss = tf.square(target - q)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))