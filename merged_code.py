import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import random
from scipy.spatial import distance
from scipy.spatial.distance import cdist

class NetworkFishSchoolSISModel:
    def __init__(self, graph_type='tree', num_fish=50, world_size=100, 
                 recovery_rate=0.1, infection_rate=0.8, fish_speed=2.0, 
                 predator_speed=3.5, predator_avoidance_weight=2.0,
                 alarm_acceleration=1.5, alarm_duration=10, predation_radius=3.0,
                 dist_threshold=0.25, loop_radius=0.2, min_clustering=0.5):
        
        """
        Initialize the fish school SIS model on a network.
        
        Parameters:
        1.) graph_type: Type of graph to use ('tree', 'loopy', or 'pruned')
        2.) num_fish: Number of fishes in the fish school
        3.) world_size: Size of the square world
        4.) recovery_rate: Probability of recovering from infected state (Probability of a fish which is now informed to go back to being uninformed )
        5.) infection_rate: Probability of getting infected when near an infected fish (probability of getting alarmed given that a connected fish in the network has been informed )
        6.) fish_speed: Base speed of fishes
        7.) predator_speed: Base Speed of the predator
        8.) predator_avoidance_weight: Weight for predator avoidance behavior
        9.) alarm_acceleration: Acceleration factor when fish get alarmed
        10.) alarm_duration: Duration of alarm acceleration in time steps
        11.) predation_radius: Radius within which fish are eaten by the predator (Implying that node becomes inactive in the network structure)
        12.) dist_threshold: Threshold for creating the full graph
        13.) loop_radius: loop radius distance so we can add triangles
        14.) min_clustering: Target minimum clustering coefficient for pruned graph
        
        """

   
        self.num_fish = num_fish
        self.world_size = world_size
        self.recovery_rate = recovery_rate
        self.infection_rate = infection_rate
        self.fish_speed = fish_speed
        self.predator_speed = predator_speed
        
        self.predator_avoidance_weight = predator_avoidance_weight
        
        self.alarm_acceleration = alarm_acceleration
        self.alarm_duration = alarm_duration
        self.predation_radius = predation_radius
        
        self.dist_threshold = dist_threshold
        self.loop_radius = loop_radius
        self.min_clustering = min_clustering
        
        # Generating graph and node positions
        self.positions = np.random.rand(num_fish, 2) * world_size
        self.pos_dict = dict(enumerate(self.positions))
        self.graph_type = graph_type
        self.generate_graph()
        
        # Initialize fish positions based on graph node positions
        self.fish_positions = np.array([self.positions[i] for i in range(num_fish)])
        
        # Initialize fish velocities (initially all are zero)
        self.fish_velocities = np.zeros((num_fish, 2))
        
        # Initialize predator position and velocity
        self.predator_position = np.array([world_size * 0.8, world_size * 0.8])
        pred_vel = np.random.uniform(-1, 1, 2)
        self.predator_velocity = pred_vel / np.linalg.norm(pred_vel) * predator_speed
        
        # Initialize infection status (0-> susceptible, 1-> infected)
        self.infection_status = np.zeros(num_fish)
        
        #how long a fish remains in accelerated state
        self.alarm_counters = np.zeros(num_fish)
        
        # Add active status (1-> active, 0-> eaten/inactive)
        self.active_status = np.ones(num_fish)
        
        # print(dir(generate_graph()))


        # Track number of eaten fish
        self.eaten_fish_count = 0
        
        # Data collection variables
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
    

    #Generating network structurures
    def generate_graph(self):

        original_threshold = self.dist_threshold
        current_threshold = original_threshold
        max_attempts = 10
        min_edges = max(self.num_fish // 2, 10)
        
        for attempt in range(max_attempts):
            G = nx.Graph()
            for i, pos in enumerate(self.positions):
                G.add_node(i, pos=pos)
            dist_matrix = cdist(self.positions, self.positions)
            
            for i in range(self.num_fish):
                for j in range(i + 1, self.num_fish):
                    if dist_matrix[i, j] < current_threshold * self.world_size:
                        G.add_edge(i, j, weight=dist_matrix[i, j])
            
            # Check if we have enough edges
            if G.number_of_edges() >= min_edges and nx.is_connected(G):
                print(f"Created graph with {G.number_of_edges()} edges using threshold {current_threshold}")
                break
            else:
                # Increase threshold for next attempt
                current_threshold *= 1.5
                print(f"Too few edges ({G.number_of_edges()}), increasing threshold to {current_threshold}")
        
        # If we still don't have enough edges, create a random connected graph
        if G.number_of_edges() < min_edges or not nx.is_connected(G):
            print("Could not create a connected graph with enough edges, creating random connected graph")
            # Create a connected random graph
            G = nx.random_tree(self.num_fish)
            # Add random edges to ensure enough connections
            edges_to_add = min_edges - G.number_of_edges()
            potential_edges = [(i, j) for i in range(self.num_fish) for j in range(i+1, self.num_fish) if not G.has_edge(i, j)]
            if potential_edges and edges_to_add > 0:
                random.shuffle(potential_edges)
                for i in range(min(edges_to_add, len(potential_edges))):
                    G.add_edge(*potential_edges[i])
        
        self.full_graph = G
        
        # Generate minimum spanning tree
        self.tree = nx.minimum_spanning_tree(self.full_graph)
        
        # Generate loopy graph
        self.loopy = self.tree.copy()
        all_edges = list(self.full_graph.edges())
        random.shuffle(all_edges)
        
        added_loops = 0
        for u, v in all_edges:
            if self.loopy.has_edge(u, v):
                continue
            u_pos, v_pos = self.positions[u], self.positions[v]
            dist = np.linalg.norm(u_pos - v_pos)
            if dist < self.loop_radius * self.world_size:
                # Look for a third node x to form a triangle
                for x in range(self.num_fish):
                    if x in (u, v):
                        continue
                    x_pos = self.positions[x]
                    if (np.linalg.norm(u_pos - x_pos) < self.loop_radius * self.world_size and
                        np.linalg.norm(v_pos - x_pos) < self.loop_radius * self.world_size):
                        self.loopy.add_edge(u, v)
                        self.loopy.add_edge(v, x)
                        self.loopy.add_edge(x, u)
                        added_loops += 1
                        break
        
        # If loopy graph has too few additional edges, add some random ones
        if self.loopy.number_of_edges() < self.tree.number_of_edges() * 1.2:
            edges_to_add = int(self.tree.number_of_edges() * 0.2)
            potential_edges = [(u, v) for u, v in all_edges if not self.loopy.has_edge(u, v)]
            if potential_edges:
                random.shuffle(potential_edges)
                for i in range(min(edges_to_add, len(potential_edges))):
                    u, v = potential_edges[i]
                    self.loopy.add_edge(u, v)
        
        # Generate pruned graph
        self.pruned = self.loopy.copy()
        extra_edges = set(self.loopy.edges()) - set(self.tree.edges())
        removed_edges = 0
        
        for u, v in list(extra_edges):
            if not self.pruned.has_edge(u, v):
                continue
            self.pruned.remove_edge(u, v)
            if not nx.is_connected(self.pruned) or nx.average_clustering(self.pruned) < self.min_clustering:
                self.pruned.add_edge(u, v)  # Restore if disconnects or too low clustering
            else:
                removed_edges += 1
        
        # Print statistics for debugging
        print(f"Graph statistics:")
        print(f"  Full graph: {self.full_graph.number_of_nodes()} nodes, {self.full_graph.number_of_edges()} edges")
        print(f"  Tree: {self.tree.number_of_nodes()} nodes, {self.tree.number_of_edges()} edges")
        print(f"  Loopy: {self.loopy.number_of_nodes()} nodes, {self.loopy.number_of_edges()} edges")
        print(f"  Pruned: {self.pruned.number_of_nodes()} nodes, {self.pruned.number_of_edges()} edges")
        
        # Set the active graph based on graph_type
        if self.graph_type == 'tree':
            self.active_graph = self.tree
        elif self.graph_type == 'loopy':
            self.active_graph = self.loopy
        elif self.graph_type == 'pruned':
            self.active_graph = self.pruned
            
        # Calculate metrics
        self.calculate_network_metrics()
    
    def calculate_network_metrics(self):
        self.metrics = {}
        
        for graph_name, graph in [('tree', self.tree), ('loopy', self.loopy), ('pruned', self.pruned)]:
            clustering = nx.average_clustering(graph)
            
            # Calculate average path length
            connected_components = list(nx.connected_components(graph))
            if connected_components:
                largest_component = max(connected_components, key=len)
                subgraph = graph.subgraph(largest_component)
                if len(subgraph) > 1:
                    try:
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    except:
                        avg_path_length = float('inf')
                else:
                    avg_path_length = 0
            else:
                avg_path_length = float('inf')
            
            # Calculate degree distribution
            degrees = [d for _, d in graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            
            self.metrics[graph_name] = {
                'clustering': clustering,
                'avg_path_length': avg_path_length,
                'avg_degree': avg_degree,
                'num_edges': graph.number_of_edges(),
                'num_nodes': graph.number_of_nodes()
            }
            
    # Visualizing the network
    def visualize_network_structure(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        graph_types = ['tree', 'loopy', 'pruned']
        graphs = [self.tree, self.loopy, self.pruned]
        colors = ['green', 'blue', 'purple']
        
        for i, (graph_type, graph, color) in enumerate(zip(graph_types, graphs, colors)):
            ax = axes[i]
            
            # Scale node positions to 0-1 range for visualization
            pos_dict = {node: self.positions[node] / self.world_size for node in graph.nodes()}
            
            # Draw the graph
            nx.draw_networkx_edges(graph, pos_dict, ax=ax, alpha=0.6, width=1.0, edge_color=color)
            nx.draw_networkx_nodes(graph, pos_dict, ax=ax, node_size=40, 
                                  node_color=color, alpha=0.7)
            
            #axis properties
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.set_title(f"{graph_type.capitalize()} Structure\n{len(graph.edges())} edges")
            
            # Adding text
            metrics = self.metrics[graph_type]
            metrics_text = (
                f"Nodes: {metrics['num_nodes']}\n"
                f"Edges: {metrics['num_edges']}\n"
                f"Avg Degree: {metrics['avg_degree']:.2f}\n"
                f"Clustering: {metrics['clustering']:.2f}\n"
                f"Avg Path Length: {metrics['avg_path_length']:.2f}"
            )
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle("Network Structure Comparison", fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network structure saved to: {save_path}")
        
        plt.show()
        return fig
    

    #Reset function to re-initialize everything
    def reset(self):
        self.infection_status = np.zeros(self.num_fish)
        self.alarm_counters = np.zeros(self.num_fish)
        self.active_status = np.ones(self.num_fish)
        self.eaten_fish_count = 0
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
    
    def change_graph_type(self, graph_type):
        if graph_type not in ['tree', 'loopy', 'pruned']:
            raise ValueError("Graph type must be 'tree', 'loopy', or 'pruned'")
        
        self.graph_type = graph_type
        if graph_type == 'tree':
            self.active_graph = self.tree
        elif graph_type == 'loopy':
            self.active_graph = self.loopy
        else:
            self.active_graph = self.pruned
    
    
    def initialize_infection(self, num_initial=1):
        # TODO: get all active fishes and initialize
        active_fish = np.where(self.active_status == 1)[0]
        if len(active_fish) > 0:
            # Choose from active fish only
            num_to_infect = min(num_initial, len(active_fish))
            initial_infected = np.random.choice(active_fish, num_to_infect, replace=False)
            self.infection_status[initial_infected] = 1
    
    def update_infection_status(self):
        """Update the infection status of all fish based on SIS model using the network structure."""
        # Current infected fish
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        
        # Find susceptible fish
        susceptible = np.where((self.infection_status == 0) & (self.active_status == 1))[0]
        
        # Check neighbors in the graph for infection spread
        newly_infected = []
        
        for s in susceptible:
            # Find all neighbors of s in the active graph
            if s in self.active_graph:  # Check if node exists in the graph
                neighbors = list(self.active_graph.neighbors(s))
                
                # Check if any neighbors are infected
                for neighbor in neighbors:
                    if neighbor in infected:
                        if np.random.random() < self.infection_rate:
                            newly_infected.append(s)
                            
                            # Set alarm counter for newly infected fish
                            self.alarm_counters[s] = self.alarm_duration
                            break
        
        # Apply infection
        self.infection_status[newly_infected] = 1
        
        # Update alarm counters
        self.alarm_counters = np.maximum(0, self.alarm_counters - 1)
        
        # Apply recovery
        for i in infected:
            if np.random.random() < self.recovery_rate:
                self.infection_status[i] = 0
        
        # Record infection count (only count active fish)
        active_infected = np.sum((self.infection_status == 1) & (self.active_status == 1))
        self.infection_history.append(active_infected)
    
    def update(self):
        """Update fish positions and infection status."""
        # Update fish movement based on network connections
        self._update_fish_velocities()
        
        # Only update positions of active fish
        active_fish = np.where(self.active_status == 1)[0]
        self.fish_positions[active_fish] += self.fish_velocities[active_fish]
        
        # Wrap around world boundaries (toroidal space)
        self.fish_positions = self.fish_positions % self.world_size
        
        # Update predator position
        self._update_predator()
        
        # Check for predation (fish being eaten)
        self._check_predation()
        
        # Check for predator-triggered infections
        self._check_predator_infection()
        
        # Update infection status based on SIS model over the network
        self.update_infection_status()
        
        # Record eaten fish count
        self.eaten_history.append(self.eaten_fish_count)
    
    def _update_fish_velocities(self):
        """Update fish velocities based on network structure and predator avoidance."""
        new_velocities = np.zeros_like(self.fish_velocities)
        
        # Only update velocities of active fish
        active_fish = np.where(self.active_status == 1)[0]
        
        for i in active_fish:
            # Find neighbors in the active graph
            if i in self.active_graph:  # Check if node exists in the graph
                neighbors = list(self.active_graph.neighbors(i))
                # Filter out inactive neighbors
                neighbors = [n for n in neighbors if self.active_status[n] == 1]
                
                if neighbors:
                    # Calculate vector towards neighbors (cohesion)
                    neighbor_positions = self.fish_positions[neighbors]
                    mean_position = np.mean(neighbor_positions, axis=0)
                    cohesion = mean_position - self.fish_positions[i]
                    if np.linalg.norm(cohesion) > 0:
                        cohesion = cohesion / np.linalg.norm(cohesion) * self.fish_speed
                    
                    # Predator avoidance
                    predator_direction = self.fish_positions[i] - self.predator_position
                    distance_to_predator = np.linalg.norm(predator_direction)
                    if distance_to_predator < self.world_size * 0.2:  # Detection radius for predator
                        if np.linalg.norm(predator_direction) > 0:
                            predator_direction = predator_direction / np.linalg.norm(predator_direction) * self.fish_speed
                            
                            # If predator is very close, trigger alarm if not already alarmed
                            if distance_to_predator < self.world_size * 0.1 and self.alarm_counters[i] == 0:
                                self.alarm_counters[i] = self.alarm_duration
                                # If not already infected, infect this fish (emergency signal)
                                if self.infection_status[i] == 0:
                                    self.infection_status[i] = 1
                    else:
                        predator_direction = np.zeros(2)
                    
                    # Add small random component to avoid stagnation
                    random_component = np.random.uniform(-1, 1, 2)
                    if np.linalg.norm(random_component) > 0:
                        random_component = random_component / np.linalg.norm(random_component) * self.fish_speed * 0.1
                    
                    # Combine all behaviors
                    velocity = cohesion + self.predator_avoidance_weight * predator_direction + random_component
                    
                    # Normalize and set speed
                    if np.linalg.norm(velocity) > 0:
                        velocity = velocity / np.linalg.norm(velocity) * self.fish_speed
                        
                        # Apply speed adjustments based on status
                        speed_multiplier = 1.0
                        
                        # If fish is infected, it moves faster
                        if self.infection_status[i] == 1:
                            speed_multiplier *= 1.5
                            
                        # If fish is in alarm state, apply additional acceleration
                        if self.alarm_counters[i] > 0:
                            speed_multiplier *= self.alarm_acceleration
                            
                        velocity *= speed_multiplier
                    
                    new_velocities[i] = velocity
                else:
                    # If no neighbors in the graph or they're all inactive, move randomly
                    random_direction = np.random.uniform(-1, 1, 2)
                    if np.linalg.norm(random_direction) > 0:
                        new_velocities[i] = random_direction / np.linalg.norm(random_direction) * self.fish_speed * 0.5
            else:
                # If node not in graph, move randomly
                random_direction = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(random_direction) > 0:
                    new_velocities[i] = random_direction / np.linalg.norm(random_direction) * self.fish_speed * 0.5
        
        self.fish_velocities = new_velocities
    
    def _update_predator(self):
        # Find the nearest active fish
        active_fish = np.where(self.active_status == 1)[0]
        if len(active_fish) == 0:
            # No active fish, move randomly
            random_dir = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(random_dir) > 0:
                self.predator_velocity = random_dir / np.linalg.norm(random_dir) * self.predator_speed
            self.predator_position += self.predator_velocity
            self.predator_position = self.predator_position % self.world_size
            return
            
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_fish_idx = active_fish[nearest_idx]
        
        # Move toward the nearest fish
        direction = self.fish_positions[nearest_fish_idx] - self.predator_position
        if np.linalg.norm(direction) > 0:
            self.predator_velocity = direction / np.linalg.norm(direction) * self.predator_speed
        
        # Update position
        self.predator_position += self.predator_velocity
        
        # Wrap around world boundaries
        self.predator_position = self.predator_position % self.world_size
    
    def _check_predation(self):
        active_fish = np.where(self.active_status == 1)[0]
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        
        # Fish that are very close to the predator get eaten
        eaten_indices = [active_fish[i] for i, d in enumerate(distances) if d < self.predation_radius]
        
        if eaten_indices:
            # Mark fish as inactive (eaten)
            self.active_status[eaten_indices] = 0
            self.eaten_fish_count += len(eaten_indices)
            
            # Remove eaten fish from infection status (no longer relevant)
            self.infection_status[eaten_indices] = 0
            
            # Log that predator has eaten fish
            print(f"Predator ate {len(eaten_indices)} fish! Total eaten: {self.eaten_fish_count}")
    
    def _check_predator_infection(self):
        predator_radius = self.predation_radius * 5  # Larger radius for detection
        
        # Only consider active fish
        active_fish = np.where(self.active_status == 1)[0]
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        
        # Fish that are close to the predator get infected and alarmed
        close_indices = [active_fish[i] for i, d in enumerate(distances) 
                         if d < predator_radius and d >= self.predation_radius]
        
        if close_indices:
            # Infect fish
            self.infection_status[close_indices] = 1
            
            # Set alarm counters
            self.alarm_counters[close_indices] = self.alarm_duration
    
    def get_statistics(self):
        active_fish = np.sum(self.active_status)
        active_infected = np.sum((self.infection_status == 1) & (self.active_status == 1))
        
        return {
            'num_active': active_fish,
            'num_infected': active_infected,
            'num_eaten': self.eaten_fish_count,
            'infection_percentage': (active_infected / active_fish * 100) if active_fish > 0 else 0,
            'eaten_percentage': (self.eaten_fish_count / self.num_fish * 100),
            'graph_metrics': self.metrics[self.graph_type]
        }
    
    def visualize(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.clear()
        
        # Plot the graph edges first
        for u, v in self.active_graph.edges():
            if self.active_status[u] == 1 and self.active_status[v] == 1:  # Only draw edges between active fish
                ax.plot([self.fish_positions[u, 0], self.fish_positions[v, 0]],
                        [self.fish_positions[u, 1], self.fish_positions[v, 1]],
                        'gray', alpha=0.3, linewidth=0.5)
        
        # Plot fish - need to account for active/inactive status
        susceptible = np.where((self.infection_status == 0) & (self.active_status == 1))[0]
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        
        # Further categorize infected fish by alarm state
        alarmed = np.where((self.infection_status == 1) & (self.active_status == 1) & (self.alarm_counters > 0))[0]
        infected_not_alarmed = np.array([i for i in infected if i not in alarmed])
        
        # Create custom colormap for eaten fish (faded gray)
        eaten = np.where(self.active_status == 0)[0]
        
        # Plot susceptible fish
        if len(susceptible) > 0:
            ax.scatter(
                self.fish_positions[susceptible, 0], 
                self.fish_positions[susceptible, 1], 
                color='blue', s=30, alpha=0.7, label='Susceptible'
            )
        
        # Plot infected fish (not alarmed)
        if len(infected_not_alarmed) > 0:
            ax.scatter(
                self.fish_positions[infected_not_alarmed, 0], 
                self.fish_positions[infected_not_alarmed, 1], 
                color='red', s=50, alpha=0.7, label='Infected'
            )
        
        # Plot alarmed fish (infected and in alarm state)
        if len(alarmed) > 0:
            ax.scatter(
                self.fish_positions[alarmed, 0], 
                self.fish_positions[alarmed, 1], 
                color='orange', s=70, alpha=0.9, label='Alarmed'
            )
            
            # Add velocity vectors for alarmed fish (show acceleration)
            ax.quiver(
                self.fish_positions[alarmed, 0],
                self.fish_positions[alarmed, 1],
                self.fish_velocities[alarmed, 0],
                self.fish_velocities[alarmed, 1],
                color='orange', scale=20, width=0.005
            )
        
        # Plot eaten fish as faded dots
        if len(eaten) > 0:
            ax.scatter(
                self.fish_positions[eaten, 0], 
                self.fish_positions[eaten, 1], 
                color='gray', s=20, alpha=0.3, label='Eaten'
            )
        
        # Plot predator
        ax.scatter(
            self.predator_position[0], 
            self.predator_position[1], 
            color='black', s=200, marker='*', label='Predator'
        )
        
        # Add predator detection radius
        predator_circle = plt.Circle(
            (self.predator_position[0], self.predator_position[1]), 
            self.predation_radius * 5, 
            color='black', fill=False, linestyle='--', alpha=0.3
        )
        ax.add_patch(predator_circle)
        
        # Add predator predation radius
        predation_circle = plt.Circle(
            (self.predator_position[0], self.predator_position[1]), 
            self.predation_radius, 
            color='red', fill=False, linestyle='--', alpha=0.5
        )
        ax.add_patch(predation_circle)
        
        # Add velocity vectors for visualization
        if len(infected) > 0:
            ax.quiver(
                self.fish_positions[infected, 0],
                self.fish_positions[infected, 1],
                self.fish_velocities[infected, 0],
                self.fish_velocities[infected, 1],
                color='red', scale=20, width=0.005
            )
        
        # Set plot limits
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(
            f'Network Fish School SIS Model ({self.graph_type.capitalize()})\n'
            f'Active: {np.sum(self.active_status)}, Infected: {np.sum((self.infection_status == 1) & (self.active_status == 1))}, '
            f'Eaten: {self.eaten_fish_count}'
        )
        
        metrics = self.metrics[self.graph_type]
        metrics_text = (
            f"Graph: {self.graph_type.capitalize()}\n"
            f"Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}\n"
            f"Avg Degree: {metrics['avg_degree']:.2f}\n"
            f"Clustering: {metrics['clustering']:.2f}\n"
            f"Avg Path Length: {metrics['avg_path_length']:.2f}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right')
        
        return ax

# Function to run the simulation with animation
def run_network_simulation(model, num_steps=100, save_animation=False, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # print(dir(ax))
    # print(dir(fig))
    
    # Initialize infection with the fish closest to the predator
    active_fish = np.where(model.active_status == 1)[0]
    distances = np.linalg.norm(model.fish_positions[active_fish] - model.predator_position, axis=1)
    closest_idx = active_fish[np.argmin(distances)]
    model.infection_status[closest_idx] = 1
    
    def update(frame):
        model.update()
        ax1 = model.visualize(ax)
        
        # Add step counter and statistics
        stats = model.get_statistics()
        stats_text = f"Step: {frame}\n"
        stats_text += f"Active Fish: {stats['num_active']}\n"
        stats_text += f"Infected: {stats['num_infected']} ({stats['infection_percentage']:.1f}%)\n"
        stats_text += f"Eaten: {stats['num_eaten']} ({stats['eaten_percentage']:.1f}%)"
        
        ax1.text(0.02, 0.78, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax1
    
    ani = FuncAnimation(fig, update, frames=range(num_steps), interval=100, blit=False)
    
    if save_animation:
        if filename is None:
            filename = f'network_fish_school_{model.graph_type}.gif'
        ani.save(filename, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

# Function to run comparative simulation on all graph types
def run_comparative_simulation(num_fish=50, num_steps=100, save_animations=False):
    results = {}
    
    for graph_type in ['tree', 'loopy', 'pruned']:
        print(f"\nRunning simulation on {graph_type} graph...")
        
        # Create model
        model = NetworkFishSchoolSISModel(
            graph_type=graph_type,
            num_fish=num_fish,
            world_size=100,
            recovery_rate=0.05,
            infection_rate=0.8,
            fish_speed=1.5,
            predator_speed=3.0,
            alarm_acceleration=1.25,
            alarm_duration=10,
            predation_radius=3.0,
            dist_threshold=0.3  # Higher threshold to ensure enough edges
        )
        
        animation = run_network_simulation(
            model, 
            num_steps=num_steps, 
            save_animation=save_animations,
            filename=f'network_fish_school_{graph_type}.gif'
        )
        
        # Store results
        results[graph_type] = {
            'infection_history': model.infection_history,
            'eaten_history': model.eaten_history,
            'metrics': model.metrics[graph_type]
        }
    
    # Plot comparative results
    plt.figure(figsize=(12, 8))
    
    # Plot infection rates over time
    plt.subplot(2, 1, 1)
    for graph_type, data in results.items():
        plt.plot(data['infection_history'], label=f'{graph_type.capitalize()}')
    plt.title('Infection Rate Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Infected Fish')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot eaten fish over time
    plt.subplot(2, 1, 2)
    for graph_type, data in results.items():
        plt.plot(data['eaten_history'], label=f'{graph_type.capitalize()}')
    plt.title('Cumulative Fish Eaten Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Fish Eaten')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparative_results.png', dpi=300)
    plt.show()
    
    # Printing network metrics
    print("\nNetwork Metrics:")
    for graph_type, data in results.items():
        print(f"\n{graph_type.capitalize()} Graph:")
        for metric, value in data['metrics'].items():
            print(f"  {metric}: {value}")
    
    return results

# Function to compare network structures
def compare_network_structures(num_fish=100, save_path=None, dist_threshold=0.3):

    model = NetworkFishSchoolSISModel(
        graph_type='tree',
        num_fish=num_fish,
        world_size=100,
        dist_threshold=dist_threshold  # Higher value creates more edges
    )
    
    # Save network structure visualization
    if save_path is None:
        save_path = 'network_structures.png'
    
    model.visualize_network_structure(save_path=save_path)
    
    return model

if __name__ == "__main__":

    model = compare_network_structures(num_fish=50, save_path='network_structures.png', dist_threshold=0.3)
    
    results = run_comparative_simulation(num_fish=50, num_steps=100, save_animations=True)