import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update the Q-value for state s and action a using the Q-learning rule.
    """
    best_next_q = np.max(Q[sprime])  # max_{a'} Q(s', a')
    td_target = r + gamma * best_next_q
    td_error = td_target - Q[s][a]

    Q[s][a] += alpha * td_error
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    Epsilon-greedy action selection:
    - with probability epsilon → random action
    - otherwise → action with max Q-value
    """
    import random

    # Exploration
    if random.random() < epsilone:
        return random.randrange(len(Q[s]))

    # Exploitation
    return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    # Q-table initialisée à 0
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01
    gamma = 0.8
    epsilon = 0.2

    n_epochs = 20
    max_itr_per_epoch = 100
    rewards = []

    for e in range(n_epochs):
        total_reward = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q, S, epsilon)

            Sprime, R, done, _, info = env.step(A)

            total_reward += R

            Q = update_q_table(Q, S, A, R, Sprime, alpha, gamma)

            S = Sprime  # Update state

            if done:
                break

        print("episode #", e, " : total reward =", total_reward)
        rewards.append(total_reward)

    print("Average reward =", np.mean(rewards))
    print("Training finished.\n")

    env.close()
