import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Load a pre-trained sentence transformer (mini but good enough)
# This model converts natural language into a fixed-size embedding vector
# It already knows about concepts like "pollution", "avoid", "move", etc.
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to turn a natural language goal into a numerical goal vector
def get_goal_embedding(goal_text):
    # Tokenize the input string for the model
    inputs = tokenizer(goal_text, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings without computing gradients (we're not training)
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling to get a single vector from the model's output
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()  # Return as a NumPy array

def describe_pollution(p):
    if p < 0.3:
        return "This area is clean."
    elif p < 0.6:
        return "This area has some pollution."
    else:
        return "This area is heavily polluted."

# Define goals as natural language strings
goal_texts = {
    "avoid pollution": "Avoid areas with high pollution while moving.",
    "prefer pollution": "Move towards the most polluted areas.",
}

# Convert each goal into a latent vector (fixed-size embedding)
# These will act as the agent's "intent", passed into its decision logic
goal_embeddings = {k: get_goal_embedding(v) for k, v in goal_texts.items()}

# STEP 2: Create a 10x10 spatial pollution map
# Each cell contains a pollution value between 0 (clean) and 1 (dirty)
# The agent will use this as part of its perception
grid_size = 10
pollution_map = np.random.rand(grid_size, grid_size)

# STEP 3: Define the agent class
class Agent:
    def __init__(self, x, y, goal_vec, label):
        self.x = x
        self.y = y
        self.goal_vec = goal_vec  # The latent goal embedding
        self.trajectory = [(x, y)]  # Keep track of movement for plotting
        self.label = label  # For display
        self.prev = None  # Previous location (to avoid backtracking)
        

    def step(self):
        # Look at the four cardinal directions and decide where to go
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_score = -float('inf')
        best_move = (0, 0)

        # Evaluate each direction
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy

            # Stay inside the grid bounds
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if self.prev == (nx, ny):
                    continue  # Don't go back to previous cell

                pollution = pollution_map[nx, ny]

                # Here's the key trick:
                # Create a dummy "pollution vector" by repeating the pollution value
                # The same shape as the goal embedding
                # This is a crude way to allow a dot product with the goal vector
                pollution_text = describe_pollution(pollution)
                pollution_vec = get_goal_embedding(pollution_text)

                # Score the move by comparing the pollution_vec and the goal_vec
                # If the goal_vec represents "avoid pollution", the dot product will be *low*
                # If the goal_vec represents "prefer pollution", the dot product will be *high*
                # This is how the agent's behavior is conditioned by its *intent*
                score = np.dot(self.goal_vec, pollution_vec)

                # Choose the move with the best score
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)

        # Remember this move
        self.prev = (self.x, self.y)
        
        # Update position
        self.x += best_move[0]
        self.y += best_move[1]
        self.trajectory.append((self.x, self.y))

# STEP 4: Create two agents with different goals
# One wants to avoid pollution, one wants to find it (for contrast)
agents = [
    Agent(0, 0, goal_embeddings["avoid pollution"], "Avoid"),
    Agent(0, grid_size - 1, goal_embeddings["prefer pollution"], "Prefer"),
]

# STEP 5: Run the simulation — each agent moves for 20 steps
for _ in range(20):
    for agent in agents:
        agent.step()

# STEP 6: Plot the results
fig, ax = plt.subplots()

# Show the pollution map as a heatmap
img = ax.imshow(pollution_map, cmap='viridis', interpolation='nearest')
fig.colorbar(img, ax=ax, label='Pollution level')

# Plot each agent's trajectory on top
colors = ['cyan', 'magenta']
for i, agent in enumerate(agents):
    xs, ys = zip(*agent.trajectory)
    ax.plot(ys, xs, marker='o', label=agent.label, color=colors[i])

ax.set_title("Agent paths with goal-conditioned behavior")
ax.legend()
plt.gca().invert_yaxis()  # Make (0,0) the top-left like in a matrix
plt.show()
