import torch
from transformers import AutoTokenizer, AutoModel

# Pick a small sentence-transformer model:
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer and model (frozen, no training)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)
encoder_model.eval()  # Put in inference mode

def encode_goal_text(goal_text: str) -> torch.Tensor:
    """
    Encode the goal text into a latent vector using a frozen language model.
    """
    inputs = tokenizer(goal_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    
    # Typical pooling strategy: mean of token embeddings
    # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
    # We'll average across seq_len
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0)  # shape [hidden_dim]


import random

class SimpleGridEnvironment:
    def __init__(self, size=5):
        self.size = size
        # Random pollution levels in each cell for demonstration
        self.grid_pollution = [[random.randint(0, 10) for _ in range(size)] 
                                for _ in range(size)]
        # Start/goal positions
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
    def get_pollution(self, x, y):
        return self.grid_pollution[x][y]
    
    def is_goal(self, x, y):
        return (x, y) == self.goal

import math

class SimpleAgent:
    def __init__(self, goal_vector: torch.Tensor):
        self.goal_vector = goal_vector  # [hidden_dim]
        
        # Hypothetical "concept embedding" that represents "pollution"
        # In a real scenario, you'd have learned or discovered these concept vectors,
        # or you might interpret them differently. This is just a demonstration.
        self.concept_pollution = torch.randn_like(self.goal_vector)
        
    def act(self, x, y, env: SimpleGridEnvironment):
        """
        Decide whether to move right or down, based on:
        - The environment's pollution levels
        - The alignment of the goal vector with the 'pollution' concept
        """
        # 1) Compute how strongly the agent's goal aligns with 'pollution' concept
        #    i.e. if the dot product is negative, it means "avoid pollution".
        #    (This is a toy heuristic. In practice, you'd do something more robust.)
        pollution_alignment = torch.dot(self.goal_vector, self.concept_pollution)
        
        # 2) Check pollution levels for "right" cell vs "down" cell (if within bounds)
        right_pollution = math.inf
        if x < env.size - 1:
            right_pollution = env.get_pollution(x+1, y)
        down_pollution = math.inf
        if y < env.size - 1:
            down_pollution = env.get_pollution(x, y+1)
        
        # 3) Decide action:
        #    If the goal aligns with avoiding pollution (i.e. negative dot product),
        #    prefer the path with lower pollution.
        #    Otherwise, pick randomly or pick the path with lower numeric index, etc.
        if pollution_alignment < 0:
            # We interpret 'pollution_alignment < 0' as "the agent cares about avoiding pollution"
            if right_pollution < down_pollution:
                return (x+1, y)
            else:
                return (x, y+1)
        else:
            # Otherwise, maybe it doesn't care about pollution, so let's just prefer going right
            # to minimize steps or something.
            if x < env.size - 1:
                return (x+1, y)
            else:
                return (x, y+1)


def run_simulation(goal_text: str):
    """
    1) Convert the text goal to a goal vector via a frozen LM encoder.
    2) Create an environment and an agent with that goal vector.
    3) Step through the environment until the agent reaches the goal.
    """
    print(f"\n=== Simulation for Goal: '{goal_text}' ===")
    
    # 1) Encode the textual goal into a vector (frozen model)
    goal_vec = encode_goal_text(goal_text)
    
    # 2) Set up environment and agent
    env = SimpleGridEnvironment(size=5)
    agent = SimpleAgent(goal_vector=goal_vec)
    
    # 3) Step through environment
    position = env.start
    path = [position]
    max_steps = 50  # just a safety limit
    
    for step in range(max_steps):
        if env.is_goal(*position):
            print(f"Reached goal at step {step}. Path: {path}")
            return
        # Agent picks next move
        next_pos = agent.act(position[0], position[1], env)
        position = next_pos
        path.append(position)
    
    print("Did not reach goal within max_steps. Path:", path)

# Let's run two simulations with different goals:
run_simulation("Avoid pollution during my commute")
run_simulation("Just get me there (I don't care about pollution)")
