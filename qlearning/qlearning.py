import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def run(iterations = 2000, isTraining = True):
    # Debug
    reward_history = np.zeros(iterations)

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="8x8",
        is_slippery=False,
        render_mode='human' if not isTraining else None
        #success_rate=1.0/3.0,
        #reward_schedule=(1, 0, 0)
    )

    learning_rate = 0.9
    discount_factor_g = 0.9
    if isTraining:
        q_table = np.zeros((env.observation_space.n, env.action_space.n)) # Rows: grid cells. Columns: rewards for each action in each cell.
    else:
        f = open("qtable.pkl", "rb")
        q_table = pickle.load(f)
        f.close

    # Epsilon Greedy learning, initially fully random
    epsilon = 1
    epsilon_decay_rate = 0.0001 # This means after 1000 iterations, agent will fully follow qtable

    for i in range(iterations):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:

            if isTraining and random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])

            new_state, reward, terminated, truncated, info = env.step(action)

            if isTraining:
                q_table[state, action] = q_table[state , action] + learning_rate * (reward + discount_factor_g * np.max(q_table[new_state,:]) - q_table[state,action])

            state = new_state

        reward_history[i] = reward
        epsilon = max(epsilon - epsilon_decay_rate,0)

        if (epsilon == 0):
            learning_rate = 0.0001 # "Stop" learning after epsilon got reduced to 0 due to decay rate

    env.close()

    # Plotting
    sum_rewards = np.zeros(iterations)
    for i in range (iterations):
        sum_rewards[i] = np.sum(reward_history[max(0, i-100):(i+1)])
    plt.plot(sum_rewards)
    plt.savefig('qlearning.png')

    # Qtable freeze
    if isTraining:
        f = open("qtable.pkl", "wb")
        pickle.dump(q_table,f)
        f.close
        

    env.close()

if __name__ == '__main__':
    run(isTraining=True, iterations=20000)