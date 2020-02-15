import random
import torch
from torch.nn.modules import Module, Conv2d, Linear, MSELoss
from torch.optim import Adam
import torch.nn.functional as fn


class Agent(Module):

    def __init__(self, state_space, channels, action_space, epsilon=0.99, epsilon_min=0.01, epsilon_decay=0.99,
                 gamma=0.9, learning_rate=0.01):
        super(Agent, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.channels = channels
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.conv1 = Conv2d(self.channels, 32, 8)
        self.conv2 = Conv2d(32, 64, 4)
        self.conv3 = Conv2d(64, 128, 3)
        self.fc1 = Linear(128 * 52 * 52, 64)
        self.fc2 = Linear(64, 32)
        self.output = Linear(32, action_space)

        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def act(self, state):
        if torch.rand(1) > self.epsilon:
            actions = self(state)
            return actions[-1].argmax().item()
        else:
            action = torch.tensor([random.randrange(self.action_space)])
        self.reduce_epsilon()
        return action

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        actions = self.output(x)
        return actions

    def learn(self, state, action, reward, next_state, done):
        q_current = self(state)[-1][action]

        if done:
            q_target = torch.tensor(reward)
        else:
            q_next = torch.max(self(next_state))
            q_target = torch.tensor(reward + (self.gamma * q_next))

        q_target.requires_grad_()

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calc_output_size(self, input_size, kernel_size, stride, padding):
        return int((input_size - kernel_size + 2 * padding) / stride) + 1

    def flatten(self, x):
        flattened_count = 1
        for dim in x.shape[1:]:
            flattened_count *= dim
        return x.view(-1, flattened_count)
