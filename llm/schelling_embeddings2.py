# ***********************************************************
# Schelling Model with Embeddings (v2)
# ***********************************************************
#
# Update from v1: households described using sentence embeddings, not bespoke variables.
#
# Created with (well, 'by' really!) ChatGPT.
# For testing ideas. I've hardly checked the code so don't know if it's right.
# Agents are described with hypothetical text descriptions that describe three features of
# each household: structure, income and political beliefs.
# These are converted to embeddings and then those embeddings are used to determine
# whether agents are happy or not.


import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import random

# -------------------------------
# Household Descriptions.
# Ten arbitrary descriptions of households, created by chatgpt, that describe their household structure, income
# and political beliefs.
# -------------------------------
household_descriptions = [
    "A dual-income, child-free couple in their 30s lives in a city loft, earns over £100k annually, and votes Green to align with their eco-conscious lifestyle.",
    "A single mother of two in a rented flat works part-time on minimum wage and supports Labour, hoping for better childcare and social welfare.",
    "A retired married couple in a suburban bungalow live comfortably on a generous pension and vote Conservative, valuing tradition and fiscal stability.",
    "A student house-share of four undergraduates lives off loans and part-time jobs, and leans toward the Liberal Democrats, favouring progressive education policy.",
    "A middle-aged, married couple with three teenagers owns a detached home in the commuter belt, earns a combined £75k, and votes Conservative for tax breaks and school choice.",
    "A cohabiting same-sex couple in their 40s living in a gentrified urban neighbourhood earn six figures and lean toward Labour for social justice and equality.",
    "A large, multi-generational family shares a terraced house in an inner-city area, has a modest combined income, and supports Labour for immigration and welfare policies.",
    "A young single professional in a high-rise flat earns £60k in tech and supports the Liberal Democrats for civil liberties and innovation.",
    "A rural, self-employed farming couple with no children earns around £40k and reliably votes Conservative, prioritising land rights and low regulation.",
    "A divorced father living part-time with his kids in a semi-detached house relies on freelance gigs and votes Green, driven by climate anxiety and local activism."
]

# -------------------------------
# Embedding Model
# -------------------------------

class EmbeddingModel:
    """Handles encoding of text descriptions using a Hugging Face transformer model."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(self._choose_device())
        self.model.to(self.device)

    def _choose_device(self):
        # Choose most suitable device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def encode(self, sentences):
        """
        Tokenizes and encodes a list of sentences into mean-pooled embeddings.
        """
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_size)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # shape: (batch, seq_len, 1)

            # Mask padding tokens, then mean-pool over the sequence length
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(1)
            counts = attention_mask.sum(1)
            mean_pooled = summed / counts

            return mean_pooled.cpu().numpy()

# -------------------------------
# Agent Class
# -------------------------------

class Agent:
    """Represents a household agent on the grid."""
    def __init__(self, desc_idx, embedding, pos):
        self.desc_idx = desc_idx
        self.embedding = embedding
        self.pos = pos
        self.happy = False

# -------------------------------
# Schelling Model Class
# -------------------------------

class SchellingModel:
    """
    Implements an embedding-based version of the Schelling segregation model.
    Agents decide to move based on similarity of text-derived embeddings.
    """
    def __init__(self, descriptions, grid_size=20, num_agents=300, similarity_threshold=0.85, max_iters=20):
        self.descriptions = descriptions
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.similarity_threshold = similarity_threshold
        self.max_iters = max_iters
        self.embedding_model = EmbeddingModel()
        self.description_embeddings = self.embedding_model.encode(self.descriptions)
        self.desc_lookup = {i: desc for i, desc in enumerate(self.descriptions)}
        self.pca = PCA(n_components=3)
        self.rgb_map = self.pca.fit_transform(self.description_embeddings)  # for RGB color plotting
        self.grid = np.full((grid_size, grid_size), None)
        self.empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        self.agents = []
        self.happy_counts = []

        self._init_agents()

    def _init_agents(self):
        """Randomly place agents on the grid with one of the household types."""
        for _ in range(self.num_agents):
            pos = random.choice(self.empty_cells)
            self.empty_cells.remove(pos)
            desc_idx = random.randint(0, len(self.descriptions) - 1)
            embedding = self.description_embeddings[desc_idx]
            agent = Agent(desc_idx, embedding, pos)
            self.grid[pos] = agent
            self.agents.append(agent)

    def _get_neighbours(self, pos):
        """Return non-empty neighbouring agents in the Moore neighbourhood."""
        x, y = pos
        neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbour = self.grid[nx, ny]
                    if neighbour is not None:
                        neighbours.append(neighbour)
        return neighbours

    def _compute_similarity(self, agent, neighbours):
        """Compute average cosine similarity between agent and neighbours."""
        if not neighbours:
            return 0
        emb_matrix = np.array([n.embedding for n in neighbours])
        sims = cosine_similarity([agent.embedding], emb_matrix)
        return np.mean(sims)

    def _get_rgb(self, desc_idx):
        """Return RGB colour (0–1 range) for a description index via PCA projection."""
        rgb = self.rgb_map[desc_idx]
        scaled = (rgb - self.rgb_map.min()) / (self.rgb_map.max() - self.rgb_map.min())
        return scaled

    def plot_grid(self, iteration):
        """Plot the current state of the grid with agent types shown by colour."""
        img = np.ones((self.grid_size, self.grid_size, 3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                agent = self.grid[i, j]
                if agent:
                    img[i, j] = self._get_rgb(agent.desc_idx)
        plt.imshow(img)
        plt.title(f"Iteration {iteration}")
        plt.axis('off')
        plt.show()

    def plot_happiness(self):
        """Plot number of happy agents per iteration."""
        plt.plot(self.happy_counts)
        plt.title("Number of Happy Agents per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Happy Agents")
        plt.show()

    def run(self):
        """Run the full simulation for the configured number of iterations."""
        for it in range(self.max_iters):
            happy = 0
            for agent in self.agents:
                neighbours = self._get_neighbours(agent.pos)
                sim = self._compute_similarity(agent, neighbours)
                if sim >= self.similarity_threshold:
                    agent.happy = True
                    happy += 1
                else:
                    agent.happy = False
                    # Move unhappy agent to a random empty cell
                    self.grid[agent.pos] = None
                    self.empty_cells.append(agent.pos)
                    new_pos = random.choice(self.empty_cells)
                    self.empty_cells.remove(new_pos)
                    agent.pos = new_pos
                    self.grid[new_pos] = agent

            self.happy_counts.append(happy)
            print(f"Iteration {it}: {happy} happy agents")
            self.plot_grid(it)

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    model = SchellingModel(household_descriptions,
                           grid_size=20,
                           num_agents=300,
                           similarity_threshold=0.53,
                           max_iters=30)
    model.run()
    model.plot_happiness()