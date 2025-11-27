import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm.
    """
    # Calcul de la valeur max pour le prochain état (Best next action)
    max_next_q = np.max(Q[sprime])
    
    # Formule du Q-Learning : Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next_q - Q[s, a])
    
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    """
    # Exploration : on choisit une action au hasard
    if np.random.uniform(0, 1) < epsilone:
        # Q.shape[1] donne le nombre d'actions (6 pour Taxi)
        return np.random.randint(0, Q.shape[1])
    
    # Exploitation : on choisit l'action avec la plus grande Q-value pour l'état s
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode=None)

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1       # Learning rate (augmenté un peu)
    gamma = 0.99      # Discount factor (standard pour Taxi)
    epsilon = 0.1     # Epsilon
    n_epochs = 2000   # Augmenté pour voir un résultat (20 est trop peu)
    max_itr_per_epoch = 100
    
    rewards = []

    print("Training started...")

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state
            S = Sprime
            
            # Stoping criteria
            if done:
                break

        # print("episode #", e, " : r = ", r) # Commenté pour éviter le spam
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    print("Training finished.\n")

    # plot the rewards in function of epochs
    plt.plot(rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.title('Evolution des récompenses')
    plt.show()

    """
    Evaluate the q-learning algorihtm
    """
    
    env_eval = gym.make("Taxi-v3", render_mode="human")
    nb_demos = 5  # Nombre de démonstrations que tu veux voir
    print(f"--- Demonstration ---")
    for i in range(nb_demos):
        S, _ = env_eval.reset()
        done = False
        total_reward = 0
           
        while not done:
            # On prend toujours la meilleure action (Exploitation pure)
            A = np.argmax(Q[S])
            
            S, R, done, _, _ = env_eval.step(A)
            
            total_reward += R
            env_eval.render() # Affiche la fenêtre graphique
            
        print(f"Score final demo {i + 1} : {total_reward}")

    env_eval.close()
    env.close()