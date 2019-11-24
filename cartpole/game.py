import torch
import gym
from statistics import mean

from cartpole.agent import Agent
from cartpole.plot import plot_averages

GYM_ID = "CartPole-v0"
MAX_SCORE = 200

env = gym.make(GYM_ID)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
snapshot_freq = 50


def play(agent: Agent, episodes):

    scores = []
    averages = []
    for episode in range(episodes):
        score = play_episode(agent)
        scores.append(score)
        avg = mean(scores)
        averages.append(avg)

        print("Episode: {} \t score: {} \t avg: {} \t epsilon: {}".format(episode + 1, score, avg, agent.epsilon))

    snapshots = take_snapshots(scores)
    title = "Cartpole - Scores over episodes for simple DQN agent using Q Learning"
    plot_averages([i * snapshot_freq for i in range(len(snapshots))], snapshots, title)


def play_episode(agent: Agent):
    score = 0

    done = False
    state = torch.tensor(env.reset(), dtype=torch.float32)
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32)

        agent.learn(state, action, reward, next_state, done)

        if score == 199:
            return MAX_SCORE

        if not done:
            state = next_state
            score += 1

    return score


def take_snapshots(scores):
    snapshots = []
    for i in range(snapshot_freq, len(scores) + 1, snapshot_freq):
        if i > (len(scores) - 1):
            snapshots.append(mean(scores[i - snapshot_freq:]))
        else:
            snapshots.append(mean(scores[i - snapshot_freq:i]))

    return snapshots


if __name__ == "__main__":
    agent = Agent(state_space, action_space)
    play(agent, 2000)
