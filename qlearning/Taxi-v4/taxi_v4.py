import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(iterations = 10000, isTraining = True):

    env = gym.make(
        'Taxi-v4',
        render_mode = 'human' if not isTraining else None
    )

    # Debug
    reward_history = np.zeros(iterations)
    success_history = np.zeros(iterations)

    # QLearning variables
    learning_rate = 0.9 # High learning rate for deterministic environment
    discount_factor = 0.9 # High propagation, reward only at the end of the execution

    if isTraining:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
            f = open("qtable.pkl", "rb")
            q_table = pickle.load(f)
            f.close()

    # Epsilon Greedy Learning
    epsilon = 1.0
    epsilon_decay_rate = 0.0001

    for i in tqdm(range(iterations)):
        state, info = env.reset()
        terminated = False
        truncated = False
        tot_reward = 0

        while not terminated and not truncated:
            if isTraining and random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                valid = np.where(info["action_mask"] == 1)[0]
                action = valid[np.argmax(q_table[state, valid])] # From valid actions, get q values, then get index of action
            

            new_state, reward, terminated, truncated, info = env.step(action)

            if isTraining:
                tot_reward+=reward
                q_table[state,action] = q_table[state,action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state,:]) - q_table[state,action])

            state = new_state

        reward_history[i] = tot_reward
        success_history[i] = (1 if terminated else 0)

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if (epsilon == 0):
            learning_rate = 0.0001

    env.close()

    
    plt.plot(reward_history)
    plt.savefig('reward_history')
    plt.clf()
    plt.plot(success_history)
    plt.savefig('success_history')

    if isTraining:
        f = open("qtable.pkl","wb")
        pickle.dump(q_table,f)
        f.close()
    


if __name__ == '__main__':
    run(iterations=10000, isTraining=False)