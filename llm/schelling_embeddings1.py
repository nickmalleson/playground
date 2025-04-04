# ***********************************************************
# Schelling Model with Embeddings (v1)
# ***********************************************************
#
# Created with (well, 'by' really!) ChatGPT.
# For testing ideas. I've hardly checked the code so don't know if it's right.
# Agents have three variables: household structure, income and political beliefs.
# These are converted to embeddings and then those embeddings are used to determine
# whether agents are happy or not.

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------------------
# 1. Agent class
# ----------------------------
class Agent:
    def __init__(self, x, y, household_structure, income, political_beliefs):
        """
        Args:
            x, y (int): Agent's location on the grid
            household_structure (str): One of ["adults with children", "single person", "house of multiple occupancy"]
            income (float): Real number in [0, 100]
            political_beliefs (str): One of ["left wing", "right wing", "populist"]
        """
        self.x = x
        self.y = y
        self.household_structure = household_structure
        self.income = income
        self.political_beliefs = political_beliefs

        # Precompute the embedding
        self.embedding = self._compute_embedding()

    def _compute_embedding(self):
        """
        Create a 7D embedding vector based on the 3 features.
        """
        # Household structure one-hot: 3D
        if self.household_structure == "adults with children":
            household_vec = [1, 0, 0]
        elif self.household_structure == "single person":
            household_vec = [0, 1, 0]
        else:  # "house of multiple occupancy"
            household_vec = [0, 0, 1]

        # Income scaled to [0, 1]
        income_val = self.income / 100.0

        # Political beliefs one-hot: 3D
        if self.political_beliefs == "left wing":
            pol_vec = [1, 0, 0]
        elif self.political_beliefs == "right wing":
            pol_vec = [0, 1, 0]
        else:  # "populist"
            pol_vec = [0, 0, 1]

        # Concatenate everything: length 7
        return np.array(household_vec + [income_val] + pol_vec)


# ----------------------------
# 2. Schelling Model
# ----------------------------
class SchellingModel:
    def __init__(self, width=20, height=20, density=0.8, similarity_threshold=0.5, max_iterations=10):
        """
        Args:
            width, height (int): Size of the grid
            density (float): Fraction of cells initially occupied by agents
            similarity_threshold (float): If the average distance to neighbors is
                                          > this threshold, the agent is unhappy.
            max_iterations (int): Number of iterations to run
        """
        self.width = width
        self.height = height
        self.density = density
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations

        # Prepare grid: None means cell is empty, otherwise store Agent
        self.grid = [[None for _ in range(self.height)] for _ in range(self.width)]

        # List of all agents
        self.agents = []

        # For tracking results
        self.history_happy_counts = []

        # Create random agents and place them
        self._initialize_agents()

        # Precompute color mapping for each agent (via 2D PCA)
        self.agent_colors = self._assign_colors_to_agents()

    def _initialize_agents(self):
        """
        Randomly populate the grid with agents, with probability = density for each cell.
        """
        # Possible categories for household structure, political beliefs
        household_options = ["adults with children", "single person", "house of multiple occupancy"]
        political_options = ["left wing", "right wing", "populist"]

        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    household = random.choice(household_options)
                    income = random.uniform(0, 100)
                    pol_belief = random.choice(political_options)

                    agent = Agent(x, y, household, income, pol_belief)
                    self.grid[x][y] = agent
                    self.agents.append(agent)

    def _assign_colors_to_agents(self):
        """
        Assign a color to each agent based on a 2D PCA projection of their 7D embeddings.
        Returns a dict: {agent_id: (r, g, b)}
        """
        embeddings = np.array([agent.embedding for agent in self.agents])
        # 2D PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(embeddings)

        # Normalize transformed coords to [0,1] for color mapping
        min_x, max_x = transformed[:,0].min(), transformed[:,0].max()
        min_y, max_y = transformed[:,1].min(), transformed[:,1].max()

        agent_colors = {}
        for i, agent in enumerate(self.agents):
            if max_x == min_x:  # Edge case if there's zero variance
                norm_x = 0.5
            else:
                norm_x = (transformed[i,0] - min_x) / (max_x - min_x)

            if max_y == min_y:
                norm_y = 0.5
            else:
                norm_y = (transformed[i,1] - min_y) / (max_y - min_y)

            # (r, g, b) from the 2D coords (just a quick way to color)
            color = (norm_x, norm_y, 0.5)
            agent_colors[agent] = color

        return agent_colors

    def _get_neighbors(self, x, y):
        """
        Return neighbors in the Moore neighborhood (up to 8 neighbors).
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[nx][ny] is not None:
                        neighbors.append(self.grid[nx][ny])
        return neighbors

    def _compute_average_distance(self, agent, neighbors):
        """
        Compute the average embedding distance between 'agent' and a list of neighbors.
        """
        if not neighbors:
            return 0.0  # No neighbors, define the distance as 0 for convenience
        distances = [np.linalg.norm(agent.embedding - nbr.embedding) for nbr in neighbors]
        return np.mean(distances)

    def step(self):
        """
        Perform one iteration of the Schelling model.
        1. For each agent, check if they're happy (average distance <= threshold).
        2. If unhappy, move them to a random empty cell.
        """
        unhappy_agents = []
        happy_count = 0

        # 1. Check who is happy/unhappy
        for agent in self.agents:
            neighbors = self._get_neighbors(agent.x, agent.y)
            avg_dist = self._compute_average_distance(agent, neighbors)
            # If avg_dist is greater than similarity_threshold => unhappy
            if avg_dist > self.similarity_threshold:
                unhappy_agents.append(agent)
            else:
                happy_count += 1

        # 2. Move unhappy agents to random empty cells
        for agent in unhappy_agents:
            # Clear old position
            self.grid[agent.x][agent.y] = None

            # Find a random empty cell
            while True:
                new_x = random.randint(0, self.width - 1)
                new_y = random.randint(0, self.height - 1)
                if self.grid[new_x][new_y] is None:
                    # Place agent there
                    agent.x = new_x
                    agent.y = new_y
                    self.grid[new_x][new_y] = agent
                    break

        return happy_count

    def run(self):
        """
        Run the full simulation for max_iterations.
        After each iteration, plot the current grid and record how many are happy.
        """
        for iteration in range(self.max_iterations):
            # Plot positions at the START of iteration
            self.plot_grid(iteration)

            # Step the model
            happy_count = self.step()
            self.history_happy_counts.append(happy_count)

        # One final plot after the last iteration
        self.plot_grid(self.max_iterations)

        # Plot a separate figure of the happy counts
        self.plot_happy_counts()

    def plot_grid(self, iteration):
        """
        Plot the grid, coloring each agent by their embedding-based color.
        """
        plt.figure()
        plt.title(f"Agent Locations at Iteration {iteration}")
        xs = []
        ys = []
        colors = []

        for agent in self.agents:
            xs.append(agent.x)
            ys.append(agent.y)
            colors.append(self.agent_colors[agent])

        plt.scatter(xs, ys, c=colors, marker='s')  # marker='s' to look more like grid cells
        plt.xlim(-1, self.width + 1)
        plt.ylim(-1, self.height + 1)
        plt.gca().invert_yaxis()  # so y=0 is at the top if desired
        plt.show()

    def plot_happy_counts(self):
        """
        Plot the number (or fraction) of happy agents vs. iteration.
        """
        plt.figure()
        plt.title("Number of Happy Agents Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Number of Happy Agents")
        plt.plot(range(len(self.history_happy_counts)), self.history_happy_counts, marker='o')
        plt.show()


# ----------------------------
# 3. Main execution
# ----------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Example usage
    model = SchellingModel(
        width=20,
        height=20,
        density=0.8,
        similarity_threshold=1.5,  # tune as desired
        max_iterations=50
    )

    model.run()