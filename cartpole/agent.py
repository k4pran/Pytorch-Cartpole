import torch
from torch.nn.modules import Module, Linear, MSELoss
from torch.optim import Adam
import torch.nn.functional as fn
import gym
import random

from cartpole.hyper_params import HyperParams


class Agent(Module):

    def __init__(self, state_space, action_space, **kwargs):
        super(Agent, self).__init__()

        self.state_space = state_space
        self.action_space = action_space

        self.epsilon = HyperParams.EPSILON.value
        self.epsilon_min = HyperParams.EPSILON_MIN.value
        self.epsilon_decay = HyperParams.EPSILON_DECAY.value
        self.gamma = HyperParams.GAMMA.value
        self.learning_rate = HyperParams.LEARNING_RATE.value

        self.override_hyper_params(kwargs)

        self.in_layer = Linear(state_space, 128)
        self.hidden_layer = Linear(128, 64)
        self.out_layer = Linear(64, action_space)

        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def override_hyper_params(self, hyper_params: dict):
        for key, val in hyper_params.items():
            if isinstance(key, str) and HyperParams.has_value(key.upper()):
                self.__setattr__(key, val)

    def forward(self, state):
        x = fn.relu(self.in_layer(state))
        x = fn.relu(self.hidden_layer(x))
        policy = self.out_layer(x)
        return policy

    def act(self, state):
        if torch.rand(1) > self.epsilon:
            action = self(state).max(0)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.action_space)]])
        self.reduce_epsilon()
        return action

    def learn(self, state, action, reward, next_state, done):
        q_current = self(state)[action]

        if done:
            q_target = torch.tensor(reward)
        else:
            q_next = torch.max(self(next_state))
            q_target = torch.tensor(reward + (self.gamma * q_next))

        q_target = q_target.unsqueeze(1)
        q_target.requires_grad_()

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    gym_id = "CartPole-v0"
    env = gym.make(gym_id)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = Agent(state_space, action_space)
    input = torch.rand(state_space)
    pred = agent(input)
