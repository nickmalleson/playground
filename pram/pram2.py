import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

n_samples = 100

with pm.Model() as model:
    # Prior distributions for the probabilities of movement
    p_left = pm.Beta('p_left', 1, 1)
    p_stay = pm.Beta('p_stay', 1, 1)
    p_right = pm.Deterministic('p_right', 1 - p_left - p_stay)

    trace = pm.sample(n_samples, chains=1)

# Take the mean of the samples as an estimate of the movement probabilities
p_left_est = np.mean(trace['p_left'])
p_stay_est = np.mean(trace['p_stay'])
p_right_est = np.mean(trace['p_right'])

# Simulate agents' movements
def simulate_movement(start_position, p_left, p_stay, p_right, n_steps=10):
    positions = [start_position]
    
    for i in range(n_steps):
        choice = np.random.choice(['left', 'stay', 'right'], p=[p_left, p_stay, p_right])
        
        if choice == 'left':
            positions.append(positions[-1] - 1)
        elif choice == 'right':
            positions.append(positions[-1] + 1)
        else:
            positions.append(positions[-1])
                
    return positions

agent1_positions = simulate_movement(0, p_left_est, p_stay_est, p_right_est)
agent2_positions = simulate_movement(10, p_left_est, p_stay_est, p_right_est)

# Visualize the results
plt.plot(agent1_positions, label='Agent 1', marker='o')
plt.plot(agent2_positions, label='Agent 2', marker='x')
plt.legend()
plt.title('Agents Movement over Time')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.show()

