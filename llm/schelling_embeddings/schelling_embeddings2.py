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
from sklearn.manifold import TSNE  # <-- ADDED
import matplotlib.colors as mcolors  # <-- ADDED

import random

# -------------------------------
# Household Descriptions.
# Some arbitrary descriptions of households, created by chatgpt, that describe their household structure, income
# and political beliefs.
# The profiles themselves aren't important as it's just to test how well the embeddings work.
# -------------------------------
household_descriptions_10 = [
    "A dual-income, child-free couple in their 30s lives in a city loft, earns over £100k annually, and votes Green to align with their eco-conscious lifestyle.",
    "A single mother of two in a rented flat works part-time on minimum wage and supports Labour, hoping for better childcare and social welfare.",
    "A retired married couple in a suburban bungalow live comfortably on a generous pension and vote Conservative, valuing tradition and fiscal stability.",
    "A student house-share of four undergraduates lives off loans and part-time jobs, and leans toward the Liberal Democrats, favouring progressive education policy.",
    "A middle-aged, married couple with three teenagers owns a detached home in the commuter belt, earns a combined £75k, and votes Conservative for tax breaks and school choice.",
    "A cohabiting same-sex couple in their 40s living in a gentrified urban neighbourhood earn six figures and lean toward Labour for social justice and equality.",
    "A large, multi-generational family shares a terraced house in an inner-city area, has a modest combined income, and supports Labour for immigration and welfare policies.",
    "A young single professional in a high-rise flat earns £60k in tech and supports the Liberal Democrats for civil liberties and innovation.",
    "A rural, self-employed farmer with no children earns around £40k and reliably votes Conservative, prioritising land rights and low regulation.",
    "A divorced father living part-time with his kids in a semi-detached house relies on freelance gigs and votes Green, driven by climate anxiety and local activism.",
    "A dual-income, couple without children in their 30s lives in a nice apartment, earn triple the median UK income, and votes for left wing parties for their progressive values.",
]

household_descriptions_20 = [
    "A dual-income, child-free couple in their 30s lives in a city loft, earns over £100k annually, and votes Green to align with their eco-conscious lifestyle.",
    "A high-income, couple without children in their 30s lives in a nice apartment, earn triple the median UK income, and votes for left wing parties for their progressive values.",
    "A single mother of two in a rented flat works part-time on minimum wage and supports Labour, hoping for better childcare and social welfare.",
    "A retired married couple in a suburban bungalow live comfortably on a generous pension and vote Conservative, valuing tradition and fiscal stability.",
    "A student house-share of four undergraduates lives off loans and part-time jobs, and leans toward the Liberal Democrats, favouring progressive education policy.",
    "A middle-aged, married couple with three teenagers owns a detached home in the commuter belt, earns a combined £75k, and votes Conservative for tax breaks and school choice.",
    "A cohabiting same-sex couple in their 40s living in a gentrified urban neighbourhood earn six figures and lean toward Labour for social justice and equality.",
    "A large, multi-generational family shares a terraced house in an inner-city area, has a modest combined income, and supports Labour for immigration and welfare policies.",
    "A young single professional in a high-rise flat earns £60k in tech and supports the Liberal Democrats for civil liberties and innovation.",
    "A rural, self-employed farming couple with no children earns around £40k and reliably votes Conservative, prioritising land rights and low regulation.",
    "A divorced father living part-time with his kids in a semi-detached house relies on freelance gigs and votes Green, driven by climate anxiety and local activism.",
    "A working-class couple with four children in a council estate earn just above minimum wage and support Labour, concerned about healthcare and public services.",
    "A wealthy family with two kids in private school lives in a five-bedroom home in the suburbs, earning over £200k, and votes Conservative for economic stability.",
    "A single pensioner living alone in a rent-controlled flat gets by on state benefits and votes Labour, worried about social care cuts.",
    "A young professional couple renting in a trendy city area earns a joint income of £90k and supports the Liberal Democrats for housing reform and progressive values.",
    "A recently arrived refugee family of five living in temporary housing survives on public assistance and supports Labour for migrant support services.",
    "An older lesbian couple who recently retired to a coastal town live modestly on pensions and vote Green for environmental protection.",
    "A middle-aged single man in a rural cottage works remotely in IT, earns £70k, and votes Liberal Democrat for balanced social and economic policies.",
    "A self-employed artist couple with one child lives in a housing co-op, earns a fluctuating income, and supports the Green Party for arts and culture funding.",
    "A traditional nuclear family with two children lives in a new-build estate, has a household income of £60k, and votes Conservative for lower taxes and strong policing.",
    "A group of migrant agricultural workers sharing accommodation earns seasonal wages and largely disengages from UK politics, though some lean Labour for workers’ rights."
]

household_descriptions = [
    "A dual-income, child-free couple in their 30s lives in a city loft, earns over £100k annually, and votes Green to align with their eco-conscious lifestyle.",
    "A single mother of two in a rented flat works part-time on minimum wage and supports Labour, hoping for better childcare and social welfare.",
    "A retired married couple in a suburban bungalow live comfortably on a generous pension and vote Conservative, valuing tradition and fiscal stability.",
    "A student house-share of four undergraduates lives off loans and part-time jobs, and leans toward the Liberal Democrats, favouring progressive education policy.",
    "A middle-aged, married couple with three teenagers owns a detached home in the commuter belt, earns a combined £75k, and votes Conservative for tax breaks and school choice.",
    "A cohabiting same-sex couple in their 40s living in a gentrified urban neighbourhood earn six figures and lean toward Labour for social justice and equality.",
    "A large, multi-generational immigrant family shares a terraced house in an inner-city area, has a modest combined income, and supports Labour for immigration and welfare policies.",
    "A young single professional in a high-rise flat earns £60k in tech and supports the Liberal Democrats for civil liberties and innovation.",
    "A rural, self-employed farming couple with no children earns around £40k and reliably votes Conservative, prioritising land rights and low regulation.",
    "A divorced father living part-time with his kids in a semi-detached house relies on freelance gigs and votes Green, driven by climate anxiety and local activism.",
    "A working-class couple with four children in a council estate earn just above minimum wage and support Labour, concerned about healthcare and public services.",
    "A wealthy family with two kids in private school lives in a five-bedroom home in the suburbs, earning over £200k, and votes Conservative for economic stability.",
    "A single pensioner living alone in a rent-controlled flat gets by on state benefits and votes Labour, worried about social care cuts.",
    "A young professional couple renting in a trendy city area earns a joint income of £90k and supports the Liberal Democrats for housing reform and progressive values.",
    "A recently arrived refugee family of five living in temporary housing survives on public assistance and supports Labour for migrant support services.",
    "An older lesbian couple who recently retired to a coastal town live modestly on pensions and vote Green for environmental protection.",
    "A middle-aged single man in a rural cottage works remotely in IT, earns £70k, and votes Liberal Democrat for balanced social and economic policies.",
    "A self-employed artist couple with one child lives in a housing co-op, earns a fluctuating income, and supports the Green Party for arts and culture funding.",
    "A traditional nuclear family with two children lives in a new-build estate, has a household income of £60k, and votes Conservative for lower taxes and strong policing.",
    "A group of migrant agricultural workers sharing accommodation earns seasonal wages and largely disengages from UK politics, though some lean Labour for workers’ rights.",
    "A widowed grandmother raising her grandchild on a state pension and child benefits votes Labour to protect social services.",
    "A wealthy singleton in their 50s living in a penthouse flat earns over £150k in finance and votes Conservative for deregulation and low taxation.",
    "A middle-income blended family with four kids in a semi-rural town earns £65k and votes Labour for education reform and family tax credits.",
    "A couple in their late 20s living in a shared ownership flat earn £50k between them and support the Greens due to concerns about the climate and housing justice.",
    "A retired army veteran and his wife live in a modest suburban home, rely on pensions, and vote Conservative out of loyalty and national pride.",
    "A household of gig-economy workers sharing a converted warehouse earns irregular income and backs Labour for employment protections.",
    "A working-class single man in a bedsit on Universal Credit feels politically alienated and does not vote.",
    "A married couple who homeschool their children live in a countryside home, earn from a small online business, and support Conservative values and independence.",
    "A high-earning consultant couple living in central London rents a luxury flat and votes Lib Dem for pro-business, pro-Europe policies.",
    "A mother and adult daughter living together in a suburban house with a household income of £35k lean Labour due to NHS concerns.",
    "A conservative Christian family with five children lives in a rural village, earns £45k from manual labour, and votes Conservative for social conservatism.",
    "A polyamorous household of five adults sharing a large rented house in the city earns mixed freelance income and backs the Greens for inclusivity and reform.",
    "A single dad working night shifts in logistics supports Labour for wage increases and safer working conditions.",
    "An elderly couple living in sheltered accommodation on fixed pensions vote Labour due to concerns about healthcare funding.",
    "A wealthy entrepreneur family living in a gated estate earns over £300k and votes Conservative for pro-business policies.",
    "A recently unemployed young couple with a baby, living with parents, support Labour for childcare and cost-of-living support.",
    "A self-sufficient off-grid couple in a rural cabin earns from crafts and online work, and votes Green for ecological resilience.",
    "A divorced woman in her 60s living in a city apartment lives on alimony and pensions, and votes Lib Dem for civil liberties.",
    "A household of musicians and baristas sharing a house in a cultural district votes Green for arts funding and sustainability.",
    "A newly naturalised citizen couple with toddlers rents in the suburbs, earns £42k jointly, and supports Labour for inclusion and opportunity."
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
        Tokenizes and encodes a list of sentences into mean-pooled embeddings (a sentence embedding that
        averages the embeddings of its tokens).

        :param sentences: A list of strings (household descriptions).
        :return: A numpy array of shape (len(sentences), hidden_size) containing
                 the mean-pooled embeddings.
        """
        print("Encoding sentences...",)
        with torch.no_grad():  # (no training so don't keep track of gradients)
            # 1) Tokenize the text
            #    - 'padding=True' ensures sequences are padded to the same length in each batch
            #    - 'truncation=True' shortens longer sequences to the model's max allowable length
            #    - 'return_tensors="pt"' outputs PyTorch tensors
            inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)

            # 2) Pass tokenized input through the model to get the final hidden states
            #    This returns an object containing:
            #      - last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            #        (the transformer output at each token position)
            outputs = self.model(**inputs)

            # 3) Extract the final hidden states for each token
            #    'outputs.last_hidden_state' gives us the representation at each sequence position
            token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

            # 4) Adjust the attention mask to match the embeddings' dimensionality
            #    The original mask is (batch_size, seq_len).
            #    We add an extra dimension so we can broadcast elementwise
            #    multiplication over the hidden_size.
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # shape: (batch_size, seq_len, 1)

            # 5) Zero out the embeddings of padding tokens.
            #    Multiplying the token embeddings with the attention mask sets any padded positions to 0.
            masked_embeddings = token_embeddings * attention_mask

            # 6) Sum embeddings across the token dimension (seq_len),
            #    effectively adding up all token vectors for each sequence in the batch.
            #    The result has shape (batch_size, hidden_size).
            summed = masked_embeddings.sum(1)

            # 7) Count how many real (non-padding) tokens each sequence has
            #    This will be needed to compute the average (mean-pooling).
            counts = attention_mask.sum(1)

            # 8) Divide the summed embeddings by the number of tokens to get the average.
            #    Each resulting embedding is now the mean over all valid (non-padding) tokens in the sequence.
            mean_pooled = summed / counts

            print("...Encoding complete.")

            # 9) Move data back to the CPU and convert to a NumPy array for further processing.
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
        self.desc_lookup = {i: desc for i, desc in enumerate(self.descriptions)}

        ## PCA for RGB mapping (so agents with similar embeddings look similar)
        #self.pca = PCA(n_components=3)
        #self.rgb_map = self.pca.fit_transform(self.description_embeddings)  # for RGB color plotting

        # ------------------------
        # COLOR MAPPING USING t-SNE (t-distributed Stochastic Neighbor Embedding)
        # ------------------------
        # 1) Project the description embeddings into 2D
        self.tsne_map = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(
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
                           similarity_threshold=0.45,
                           max_iters=100)
    model.run(do_plots=True)
    model.plot_happiness(return_fig=False)