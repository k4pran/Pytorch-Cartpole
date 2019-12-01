from collections import deque
import random
import torch

from cartpole.agent import Agent
from cartpole.hyper_params import HyperParams


class ReplayMemory:

    def __init__(self):
        self.memory = deque(maxlen=HyperParams.MEMORY_MAX.value)

    def memorise(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def memory_count(self):
        return len(self.memory)


class ReplayAgent(Agent):

    def __init__(self, state_space, action_space, **kwargs):
        super().__init__(state_space, action_space, **kwargs)
        self.batch_Size = HyperParams.BATCH_SIZE.value
        self.override_hyper_params(kwargs)
        self.memory = ReplayMemory()

    def memorise(self, state, action, reward, next_state, is_terminal):
        self.memory.memorise(zip([state, action, reward, next_state, is_terminal]))

    def learn(self, state, action, reward, next_state, is_terminal):
        self.memory.memorise((
            state,
            action,
            reward,
            next_state,
            is_terminal
        ))

        if self.memory.memory_count() < self.batch_Size:
            return

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_is_terminal \
            = zip(*self.memory.sample(self.batch_Size))

        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_states)), dtype=torch.bool)

        batch_states = torch.stack(batch_states)
        batch_actions = torch.cat(batch_actions)
        batch_rewards = torch.cat(batch_rewards)
        batch_next_states = torch.stack([s for s in batch_next_states if s is not None])

        batch_q_current = self(batch_states).gather(1, batch_actions)

        batch_q_next = torch.zeros(self.batch_Size)
        batch_q_next[non_terminal_mask] = self(batch_next_states).max(1)[0].detach()

        q_target = torch.tensor(batch_rewards + (self.gamma * batch_q_next)).unsqueeze(1)
        q_target.requires_grad_()

        loss = self.loss_fn(batch_q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


