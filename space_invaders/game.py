import random
from collections import deque

import gym
import torch
from torchvision.transforms import transforms

from space_invaders.agent import Agent

GYM_ID = "SpaceInvaders-v0"
EPISODES = 1000

env = gym.make(GYM_ID)
state_space = env.observation_space.shape
action_space = env.action_space.n
image_dims = (64, 64)
frames_nb = 3


def play(agent, render=False):

    for episode in range(EPISODES):
        done = False
        frames = deque(maxlen=frames_nb)
        [frames.append(state) for state in init_first_frames()]
        score = 0
        while not done:

            if (render):
                env.render()

            stacked_state = stack_frames(frames)
            action = agent.act(stacked_state)
            next_state, reward, done, _ = env.step(action)

            next_state = prepare_state_tensor(next_state)

            state = next_state
            frames.append(state)
            stacked_next_state = stack_frames(frames)

            agent.learn(stacked_state, action, reward, stacked_next_state, done)

            score += reward
            print(score)

        print("Episode {} -- score: ".format(episode, score))


def init_first_frames():
    frames = [prepare_state_tensor(env.reset())]
    for i in range(frames_nb - 1):
        action = random.randrange(action_space)
        state, _, _, _ = env.step(action)
        state = prepare_state_tensor(state)
        frames.append(state)

    return frames


def prepare_state_tensor(state):
    transform = get_transform()
    state = torch.tensor(state, dtype=torch.float32)
    state = transform(state.permute(2, 0, 1)).squeeze()
    state = state.reshape((1, 64, 64))

    return state


def stack_frames(frames):
    return torch.stack(tuple(frames))


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_dims),
        transforms.ToTensor(),
    ])


if __name__ == "__main__":
    agent = Agent(image_dims, 1, action_space)
    play(agent)