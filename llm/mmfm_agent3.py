import torch
from transformers import AutoTokenizer, AutoModel
import random
import math
import matplotlib.pyplot as plt

#############################
# 1) Pre-trained (frozen) sentence encoder
#############################
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)
encoder_model.eval()  # inference mode (no training)

def encode_text(text: str) -> torch.Tensor:
    """
    Encode text into a latent vector using a frozen sentence-transformer model.
    We'll do simple mean pooling of the token embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # shape [1, hidden_dim]
    return embeddings.squeeze(0)  # shape [hidden_dim]

#############################
# 2) A Simple 5×5 Grid Environment
#############################
class SimpleGridEnvironment:
    def __init__(self, size=5):
        self.size = size
        # Random pollution levels in each cell
        self.grid_pollution = [
            [random.randint(0, 10) for _ in range(size)]
            for _ in range(size)
        ]
        # Start & goal positions
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        
    def get_pollution(self, x, y):
        return self.grid_pollution[x][y]
    
    def is_goal(self, x, y):
        return (x, y) == self.goal

#############################
# 3) Agent That Decides Based on Goal Similarity
#############################
class PollutionAwareAgent:
    def __init__(self, goal_text: str):
        """
        - goal_text: e.g. "avoid pollution" or "get there fast"
        """
        self.goal_vector = encode_text(goal_text)
        
        # Define a concept vector for 'pollution'
        # We'll use the same encoder to keep everything in the same embedding space
        self.concept_pollution = encode_text("pollution")
        
        # We'll precompute the cosine similarity to see if the agent's goal
        # is aligned with the idea of 'pollution' (meaning it *cares* about it).
        self.pollution_similarity = self.cosine_similarity(
            self.goal_vector, self.concept_pollution
        )
        
    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=1
        ).item()
    
    def act(self, x, y, env: SimpleGridEnvironment):
        """
        Decide whether to move RIGHT or DOWN, comparing pollution levels
        if the agent "cares" about avoiding pollution.
        """
        right_pollution = math.inf
        if x < env.size - 1:
            right_pollution = env.get_pollution(x+1, y)
        down_pollution = math.inf
        if y < env.size - 1:
            down_pollution = env.get_pollution(x, y+1)
        
        # If the goal is strongly related to pollution (above a threshold),
        # we interpret that as "the agent wants to avoid pollution."
        # => choose the path with lower pollution.
        threshold = 0.2  # Adjust as you like. Higher means stricter "care" about pollution
        if self.pollution_similarity > threshold:
            # Agent is "avoiding pollution": pick path with lower pollution
            if right_pollution < down_pollution and right_pollution != math.inf:
                return (x+1, y)
            elif down_pollution != math.inf:
                return (x, y+1)
            else:
                return (x, y)  # no move if stuck
            
        else:
            # If the agent doesn't "care," just move right if possible, else down
            if x < env.size - 1:
                return (x+1, y)
            elif y < env.size - 1:
                return (x, y+1)
            else:
                return (x, y)

#############################
# 4) Run Simulation + Plot
#############################
def run_simulation(goal_text: str, show_plot=True):
    """
    1) Initialize environment & agent with the given goal text.
    2) Step through the environment, logging the path.
    3) Plot the pollution grid + path on a 2D chart.
    """
    print(f"\n=== Simulation for Goal: '{goal_text}' ===")
    
    env = SimpleGridEnvironment(size=5)
    agent = PollutionAwareAgent(goal_text=goal_text)
    
    # Print out the agent's pollution similarity for debugging
    print(f"Cosine similarity between goal='{goal_text}' and 'pollution': "
          f"{agent.pollution_similarity:.3f}")
    
    position = env.start
    path = [position]
    max_steps = 50
    
    for step in range(max_steps):
        if env.is_goal(*position):
            print(f"Reached goal at step {step}. Path: {path}")
            break
        next_pos = agent.act(position[0], position[1], env)
        if next_pos == position:
            # Agent didn't move. Possibly stuck at boundary or something else.
            print(f"Agent can't move further. Stopping. Path: {path}")
            break
        position = next_pos
        path.append(position)
    else:
        print("Did not reach goal within max_steps.")
    
    # Visualize
    if show_plot:
        plt.figure()
        plt.title(f"Route for Goal: '{goal_text}'")
        # Show pollution as an image
        plt.imshow(env.grid_pollution, origin='upper')
        
        # Plot path
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        plt.plot(cols, rows, marker='o')  # row=x => y-axis, col=y => x-axis
        
        plt.text(0, 0, "Start", ha='left', va='top')
        plt.text(env.size - 1, env.size - 1, "Goal", ha='right', va='bottom')
        
        plt.colorbar(label="Pollution Level")  # optional color scale
        plt.show()
    
    return path

#############################
# 5) Demonstration
#############################

# Example 1: The agent explicitly tries to avoid pollution
_ = run_simulation("avoid pollution")

# Example 2: The agent doesn't mention pollution
_ = run_simulation("just get me to the end")
