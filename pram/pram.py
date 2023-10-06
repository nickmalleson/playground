import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
num_agents = 2 
max_x = 10
max_y = 10

# Initialize random walk step size distributions
step_size = pm.Uniform('step_size', lower=0, upper=2)
angle = pm.VonMises('angle', mu=0, kappa=0.1) 

# Initialize starting positions
start_pos = np.array([[1, 1], [5, 5]])
pos = pm.Deterministic('pos', start_pos)

# Step function for random walk
@pm.deterministic
def walk(pos=pos, step=step_size, angle=angle):
    dir_x = np.cos(angle) * step
    dir_y = np.sin(angle) * step
    return pos + np.array([dir_x, dir_y]).T

# Build model 
model = pm.Model()
with model:

    # Empty step for starting positions
    pm.Deterministic('step_0', pos)

    # Multiple steps in the walk
    for i in range(100):
        pos = walk(pos=pos)

        # Bounce off edges
        pos = pm.math.maximum(0, pos)
        pos = pm.math.minimum(max_x, pos[:,0])
        pos = pm.math.minimum(max_y, pos[:,1])

        # Record position at each step
        pm.Deterministic('step_{}'.format(i+1), pos) 

# Sample from model
trace = pm.sample(1000)

# After sampling
traces = pm.trace.MultiTrace(trace) 

# Extract traces for plot
x1 = traces['step_0'][:,0,0]
y1 = traces['step_0'][:,0,1] 
x2 = traces['step_0'][:,1,0]
y2 = traces['step_0'][:,1,1]


# Plot samples for each agent position over time
plt.plot(x1, y1, alpha=0.5)
plt.plot(x2, y2, alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Agent Random Walk Samples')
plt.xlim(0,10) 
plt.ylim(0,10)

plt.show()
