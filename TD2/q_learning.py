"""
Q-learning on Taxi-v3.

- State definition: (taxi_row, taxi_col, passenger_loc_or_in_taxi, destination)
  environment encodes states as integers in [0, 499].
- Actions: 6 discrete actions (South, North, East, West, Pickup, Dropoff)
- Rewards: built-in Taxi rewards as described in the problem statement.

"""

import numpy as np
import random
import gymnasium as gym

import matplotlib.pyplot as plt
from collections import deque

# ---------------------------
# Q-learning update function
# ---------------------------
def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update the Q-value for state s and action a using the Q-learning rule.
    """
    best_next_q = np.max(Q[sprime])  # max_{a'} Q(s', a')
    td_target = r + gamma * best_next_q
    td_error = td_target - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q

# ---------------------------
# Epsilon-greedy policy
# ---------------------------
def epsilon_greedy(Q, s, epsilon):
    """
    Epsilon-greedy action selection:
    - with probability epsilon → random action
    - otherwise → action with max Q-value
    """
    if random.random() < epsilon:
        return random.randrange(Q.shape[1])
    return np.argmax(Q[s])

# ---------------------------
# Training script
# ---------------------------
def train_taxi_qlearning(
        env_name="Taxi-v3",
        n_episodes=20000,
        max_steps_per_episode=200,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        render_every=None   # if integer, show render every N episodes at the end of episode
    ):
    env = gym.make(env_name)  # use default render_mode for training (no gui)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    epsilon = epsilon_start
    rewards_per_episode = []
    moving_avg_window = 200
    moving_avg = []

    best_avg = -np.inf
    last_print = 0

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset() if "reset" in env.reset.__name__ else env.reset()  # compatibility gym/gymnasium
        total_reward = 0

        for t in range(max_steps_per_episode):
            action = epsilon_greedy(Q, state, epsilon)
            step_result = env.step(action)
            # compatibility with gym vs gymnasium step return shapes
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            total_reward += reward

            # Q update
            Q = update_q_table(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state
            if done:
                break

        # decaying epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # moving average for monitoring
        if len(rewards_per_episode) >= moving_avg_window:
            ma = np.mean(rewards_per_episode[-moving_avg_window:])
        else:
            ma = np.mean(rewards_per_episode)
        moving_avg.append(ma)

        # occasional logging
        if ep % 500 == 0 or ep == 1:
            print(f"Episode {ep}/{n_episodes}  |  reward={total_reward:.1f}  |  eps={epsilon:.4f}  |  MA({moving_avg_window})={ma:.2f}")

        # Optionally render the environment at intervals (one episode render)
        if render_every and ep % render_every == 0:
            # render the last episode (create render env)
            try:
                render_env = gym.make(env_name, render_mode="human")
                s, _ = render_env.reset()
                done = False
                render_env.render()
                while not done:
                    a = np.argmax(Q[s])
                    s, r, terminated, truncated, _ = render_env.step(a)
                    render_env.render()
                    done = terminated or truncated
                render_env.close()
            except Exception as exc:
                print("Rendering failed (maybe no display). Exception:", exc)

    env.close()
    return Q, rewards_per_episode, moving_avg

# ---------------------------
# Plotting helper
# ---------------------------
def plot_rewards(rewards, moving_avg):
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label="Episode reward")
    plt.plot(moving_avg, label="Moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Taxi-v3 Q-learning training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ---------------------------
# Demo: run policy
# ---------------------------
def demo_policy(Q, env_name="Taxi-v3", n_episodes=5, max_steps=200, sleep=0.5):
    import time
    try:
        env = gym.make(env_name, render_mode="human")
    except TypeError:
        # older gym may not accept render_mode on creation; create and call render()
        env = gym.make(env_name)
    for ep in range(n_episodes):
        state, _ = env.reset() if "reset" in env.reset.__name__ else env.reset()
        done = False
        total_reward = 0
        steps = 0
        # render initial state if possible
        try:
            env.render()
        except Exception:
            pass

        while not done and steps < max_steps:
            action = np.argmax(Q[state])
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result
            total_reward += reward
            steps += 1
            try:
                env.render()
                time.sleep(sleep)
            except Exception:
                pass

        print(f"Demo episode {ep+1}: total_reward={total_reward}, steps={steps}")

    try:
        env.close()
    except Exception:
        pass

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Hyperparameters
    n_episodes = 20000           # increase for better policy
    alpha = 0.1
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9996

    # Train
    Q, rewards, moving_avg = train_taxi_qlearning(
        env_name="Taxi-v3",
        n_episodes=n_episodes,
        max_steps_per_episode=200,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        render_every=None  # set to an int like 1000 to visualize periodically
    )

    # Print final stats
    print("Training finished.")
    print(f"Average reward (last 1000 episodes): {np.mean(rewards[-1000:]):.2f}")

    # Plot rewards and moving average
    plot_rewards(rewards, moving_avg)

    # Demo greedy policy for a few episodes (renders if available)
    demo_policy(Q, env_name="Taxi-v3", n_episodes=5, max_steps=200, sleep=0.2)
