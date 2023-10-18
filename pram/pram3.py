import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt


# Parameters
grid_size = 50
n_steps = grid_size  
n_samples = 100

# Probabilistic Model
with pm.Model() as model:
    # Use Dirichlet to ensure probabilities sum to 1
    movement_probs = pm.Dirichlet('movement_probs', a=np.array([5, 2, 2, 2, 2]))
    
    trace = pm.sample(n_samples, chains=1)

# Take mean of samples as the estimated probabilities
prob_values = {
    'right': np.mean(trace['movement_probs'][:, 0]),
    'left': np.mean(trace['movement_probs'][:, 1]),
    'up': np.mean(trace['movement_probs'][:, 2]),
    'down': np.mean(trace['movement_probs'][:, 3]),
    'stay': np.mean(trace['movement_probs'][:, 4])
}

def simulate_movement(start, probabilities, n_steps):
    x, y = start
    trace = [(x, y)]
    
    for _ in range(n_steps):
        move = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        if move == 'right': x += 1
        elif move == 'left': x -= 1
        elif move == 'up': y += 1
        elif move == 'down': y -= 1
        
        trace.append((x, y))
    
    return trace

# Simulate agent movements
agent1_trace = simulate_movement((0, np.random.randint(0, grid_size)), prob_values, n_steps)
agent2_trace = simulate_movement((0, np.random.randint(0, grid_size)), prob_values, n_steps)

# Create a grid to capture agent traces
grid = np.zeros((grid_size, grid_size))

for x, y in agent1_trace:
    if 0 <= x < grid_size and 0 <= y < grid_size:
        grid[y, x] += 1

for x, y in agent2_trace:
    if 0 <= x < grid_size and 0 <= y < grid_size:
        grid[y, x] += 1

# Normalize grid for visualization
grid = grid / n_steps

# Visualization
plt.imshow(grid, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label='Probability')
plt.title('Probabilistic Traces of Agents')
plt.show()

