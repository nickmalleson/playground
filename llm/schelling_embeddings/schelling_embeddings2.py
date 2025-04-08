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
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer



# -------------------------------
# Household Descriptions.
# Some arbitrary descriptions of households, created by an LLM, that describe their household structure, income
# and political beliefs.
# The profiles themselves aren't important as it's just to test how well the embeddings work.
# Note that when the model runs in earnest (see schelling_embeddinggs.ipynb) it reads a large number of descriptions
# -------------------------------
household_descriptions = [
    "A retired couple living alone in a semi-detached house in a suburban area, relying on state pensions and modest savings, strongly supporting the Conservative party",
    "An elderly couple residing in a suburban, semi-detached house, drawing income from their savings and their state pensions, voting for the Conservative party consistently",
    "A young, single professional renting a studio flat in a city centre, earning a salary around £35,000 from a career in marketing, voting for the Liberal Democrats and actively campaigning for environmental causes",
    "A large, multi-generational family residing in a terraced house, with the patriarch working as a manual labourer on a zero-hours contract, the matriarch a part-time carer, and several children, identifying as Labour supporters and strongly union-backed",
    "A single parent with three children, living in a council flat, surviving on a tight budget that includes Universal Credit and Child Tax Credits, and staunchly supporting the Labour party, particularly its more left-wing elements",
]

# -------------------------------
# Embedding Model
# -------------------------------

class EmbeddingModel:
    """Handles encoding of text descriptions using a Hugging Face transformer model."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModel.from_pretrained(model_name)
        #self.model.eval()
        #self.device = torch.device(self._choose_device())
        #self.model.to(self.device)

    def _choose_device(self):
        """
        Choose the most suitable device.
        DEPRECATED: using SentenceTransformer now.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def encode(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings


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
        # Descriptions of the generic households
        self.descriptions = descriptions

        # Set up the model
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), None)
        self.num_agents = num_agents
        self.similarity_threshold = similarity_threshold
        self.max_iters = max_iters
        self.empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        self.agents = []
        self.happy_counts = []

        # Calculate the embeddings for the household descriptions
        self.embedding_model = EmbeddingModel()
        self.description_embeddings = self.embedding_model.encode(self.descriptions)
        print(f"Embedding shape: {self.description_embeddings.shape}")
        self.desc_lookup = {i: desc for i, desc in enumerate(self.descriptions)}

        ## PCA for RGB mapping (so agents with similar embeddings look similar)
        #self.pca = PCA(n_components=3)
        #self.rgb_map = self.pca.fit_transform(self.description_embeddings)  # for RGB color plotting

        # ------------------------
        # COLOR MAPPING USING t-SNE (t-distributed Stochastic Neighbor Embedding)
        # ------------------------
        # 1) Project the description embeddings into 2D
        perplexity = min(5, len(self.description_embeddings) - 1)  # 5, or fewer if there are < 5 samples
        self.tsne_map = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(
            self.description_embeddings
        )

        # 2) Normalize into [0,1] so we can map to HSV
        t_min = self.tsne_map.min(axis=0)
        t_max = self.tsne_map.max(axis=0)
        tsne_norm = (self.tsne_map - t_min) / (t_max - t_min + 1e-8)

        # 3) Convert each point to HSV -> RGB
        #    We'll treat tsne_norm[:,0] as 'hue' and tsne_norm[:,1] as 'value' for variety.
        #    Keep saturation high, e.g. 0.8
        self.rgb_map = []
        for i in range(len(tsne_norm)):
            hue = tsne_norm[i, 0]      # 0 to 1
            saturation = 0.8
            value = 0.9 - 0.4 * tsne_norm[i, 1]  # vary from about 0.9 down to 0.5
            # hsv_to_rgb expects (h, s, v)
            color_rgb = mcolors.hsv_to_rgb((hue, saturation, value))
            self.rgb_map.append(color_rgb)

        self.rgb_map = np.array(self.rgb_map)

        # Initialize the grid with agents
        self._init_agents()

    def _init_agents(self):
        """Randomly place agents on the grid with one of the household types."""
        for _ in range(self.num_agents):
            # Add the agent to the grid
            pos = random.choice(self.empty_cells)
            self.empty_cells.remove(pos)
            # Define the agent 'type' (from the descriptions)
            desc_idx = random.randint(0, len(self.descriptions) - 1)
            embedding = self.description_embeddings[desc_idx]
            # Create the agent
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
        # Put the neighbour's embeddings into a matrix
        emb_matrix = np.array([n.embedding for n in neighbours])
        # Calculate the cosine similarities between the agent and all neighbours
        sims = cosine_similarity([agent.embedding], emb_matrix)
        # Return the mean similarity
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

    def plot_happiness(self, return_fig=False):
        """Plot number of happy agents per iteration. Return the plot if requested."""
        fig, ax = plt.subplots()
        ax.plot(self.happy_counts)
        ax.set_title("Number of Happy Agents per Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Happy Agents")
        if return_fig:
            return fig
        else:
            plt.show()

    def run(self, do_plots=True):
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
            if do_plots:
                self.plot_grid(it)

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    model = SchellingModel(household_descriptions,
                           grid_size=20,
                           num_agents=300,
                           similarity_threshold=0.65,
                           max_iters=50)
    model.run(do_plots=True)
    model.plot_happiness(return_fig=False)