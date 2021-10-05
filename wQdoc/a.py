# Load OpenAI Gym and other necessary packages.
import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt


def terminal_state(dest_idx):
    ter_taxi_row = (dest_idx // 2) * 4
    ter_taxi_col = (dest_idx % 2) * 4

    return env.encode(ter_taxi_row, ter_taxi_col, dest_idx, dest_idx)


# Environment.
env = gym.make("Taxi-v3")

# Training parameters for Q learning.
alpha = 0.9  # Learning rate.
gamma = 0.9  # Future reward discount factor.
num_of_episodes = 1000
num_of_steps = 500  # per each episode.

# Q tables for rewards.
Q_reward = 0 * np.ones((500, 6))  # 500 states and six actions.

# All possible terminal states.
dest_loc = [(0, 0), (0, 4), (4, 0), (4, 3)]
ter_states = []
for dest_idx in range(len(dest_loc)):
    ter_states.append(env.encode(dest_loc[dest_idx][0], dest_loc[dest_idx][1], dest_idx, dest_idx))

Q_reward[ter_states, :] = np.zeros((4, 6))

# Training w/ random sampling of actions.

for episode in range(num_of_episodes):
    cur_state = env.reset()

    for step in range(num_of_steps):

        action = np.argmax(Q_reward[cur_state, :])

        action = np.random.randint(0, 6)
        next_state, reward, done, info = env.step(action)

        # print(next_state, cur_state, action)
        if next_state == cur_state:
            action = np.random.randint(0, 6)
            next_state, reward, done, info = env.step(action)

        Q_reward[cur_state, action] += alpha * (reward +
                                            gamma * np.max(Q_reward[next_state, :]) - Q_reward[cur_state, action])

        if done:
            break
        else:
            cur_state = next_state

test_time = 1000
num_of_action = np.zeros(test_time)
tot_reward = np.zeros(test_time)

for test in range(test_time):
    state = env.reset()
    for t in range(50):
        action = np.argmax(Q_reward[state, :])
        state, reward, done, info = env.step(action)

        num_of_action[test] += 1
        tot_reward[test] += reward

        # For display purpose.
        # env.render()
        # time.sleep(0.5)

        if done or t == 49:
            print(f"Test time: {test}. Total reward: {tot_reward[test]}")
            break


plt.plot(num_of_action, tot_reward, 'o')
plt.title(f"Numbers of actions of {test_time} running time and respected total rewards")
plt.show()

print(f"Mean of total reward of {test_time} running time: {np.mean(tot_reward)}")
