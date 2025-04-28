import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import random
from scipy.spatial import distance

class TreeNetworkSISModel:
    def __init__(self, num_agents=100, world_size=100, infection_radius=10, 
                 recovery_rate=0.1, infection_rate=0.8, max_speed=1.0, 
                 predator_speed=2.5, tree_depth=3, branching_factor=3, 
                 position_jitter=5.0, predator_avoidance_weight=2.0,
                 alarm_acceleration=2.0, alarm_duration=10, predation_radius=3.0):

        self.num_agents = num_agents
        self.world_size = world_size
        self.infection_radius = infection_radius
        self.recovery_rate = recovery_rate
        self.infection_rate = infection_rate
        self.max_speed = max_speed
        self.predator_speed = predator_speed
        self.tree_depth = tree_depth
        self.branching_factor = branching_factor
        self.position_jitter = position_jitter
        self.predator_avoidance_weight = predator_avoidance_weight
        
        self.alarm_acceleration = alarm_acceleration
        self.alarm_duration = alarm_duration
        self.predation_radius = predation_radius
        
        self.tree = nx.balanced_tree(branching_factor, tree_depth)
        
        if self.tree.number_of_nodes() > num_agents:
            self._truncate_tree(num_agents)
        
        # If the tree is too small, add random nodes
        while self.tree.number_of_nodes() < num_agents:
            self._add_random_node()
            
        self.tree = nx.convert_node_labels_to_integers(self.tree)
        
        # Calculate positions in a tree layout for visualization
        self.positions = self._calculate_tree_positions()
        
        # Add some random movement to make it more natural
        self.positions += np.random.normal(0, position_jitter, (num_agents, 2))
        
        # Keep positions within world bounds
        self.positions = np.clip(self.positions, 0, world_size)
        
        # Initialize velocities
        self.velocities = np.zeros((num_agents, 2))
        
        # Initialize infection status (0: susceptible, 1: alarmed)
        self.infection_status = np.zeros(num_agents)
        
        # Add alarm counters (how long an agent remains in accelerated state)
        self.alarm_counters = np.zeros(num_agents)
        
        # Add active status (1: active, 0: eaten/inactive)
        self.active_status = np.ones(num_agents)
        
        # Track number of eaten agents
        self.eaten_count = 0
        
        # Network representation (start with the tree structure)
        self.graph = self.tree.copy()
        
        # Data collection
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
        
        # Define predator (threat) position
        self.predator_position = np.array([world_size * 0.8, world_size * 0.8])
        pred_vel = np.random.uniform(-1, 1, 2)
        self.predator_velocity = pred_vel / np.linalg.norm(pred_vel) * predator_speed
        
    def _truncate_tree(self, target_size):
        while self.tree.number_of_nodes() > target_size:
            leaf_nodes = [node for node, degree in self.tree.degree() if degree == 1]
            if not leaf_nodes:
                break
            self.tree.remove_node(random.choice(leaf_nodes))
            
    def _add_random_node(self):
        existing_nodes = list(self.tree.nodes())
        parent = random.choice(existing_nodes)
        new_node = max(existing_nodes) + 1
        self.tree.add_edge(parent, new_node)
            
    def _calculate_tree_positions(self):
        """Calculate positions for nodes in a hierarchical tree layout."""
        pos = nx.spring_layout(self.tree, k=2.0, iterations=100, seed=42)
        
        self._make_more_hierarchical(pos)
        
        positions = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            if i in pos:
                x, y = pos[i]
                x = (x + 1) / 2 * self.world_size * 0.8 + self.world_size * 0.1
                y = (y + 1) / 2 * self.world_size * 0.8 + self.world_size * 0.1
                positions[i] = [x, y]
        
        return positions
        
    def _make_more_hierarchical(self, pos):
        root = 0
            
        levels = {node: -1 for node in self.tree.nodes()}
        levels[root] = 0
        queue = [root]
        while queue:
            node = queue.pop(0)
            level = levels[node]
            for neighbor in self.tree.neighbors(node):
                if levels[neighbor] == -1:  # Not visited
                    levels[neighbor] = level + 1
                    queue.append(neighbor)
        
        max_level = max(levels.values())
        for node, level in levels.items():
            normalized_level = -0.8 + 1.6 * (level / max_level if max_level > 0 else 0)
            pos[node] = (pos[node][0], normalized_level)
    
    def reset(self):
        self.infection_status = np.zeros(self.num_agents)
        self.alarm_counters = np.zeros(self.num_agents)
        self.active_status = np.ones(self.num_agents)
        self.eaten_count = 0
        self.graph = self.tree.copy()
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
        
    def initialize_infection(self, num_initial=1):
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) > 0:
            num_to_infect = min(num_initial, len(active_agents))
            initial_infected = np.random.choice(active_agents, num_to_infect, replace=False)
            self.infection_status[initial_infected] = 1
            self.alarm_counters[initial_infected] = self.alarm_duration
    
    def update_infection_status(self):
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        
        newly_infected = []
        
        for i in infected:
            neighbors = list(self.tree.neighbors(i))
            
            susceptible_neighbors = [n for n in neighbors if self.infection_status[n] == 0 and self.active_status[n] == 1]
            
            for s in susceptible_neighbors:
                if np.random.random() < self.infection_rate:
                    newly_infected.append(s)
                    self.graph.add_edge(i, s)
                    self.alarm_counters[s] = self.alarm_duration
        
        self.infection_status[newly_infected] = 1
        
        self.alarm_counters = np.maximum(0, self.alarm_counters - 1)
        
        for i in infected:
            if np.random.random() < self.recovery_rate:
                self.infection_status[i] = 0
        
        active_infected = np.sum((self.infection_status == 1) & (self.active_status == 1))
        self.infection_history.append(active_infected)
    
    def compute_network_metrics(self):
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) == 0:
            return None
            
        active_graph = self.graph.subgraph(active_agents)
        
        if len(active_graph.edges) > 0:
            try:
                clustering = nx.average_clustering(active_graph)
            except:
                clustering = 0
            
            connected_components = list(nx.connected_components(active_graph))
            if connected_components:
                largest_component = max(connected_components, key=len)
                subgraph = active_graph.subgraph(largest_component)
                if len(subgraph) > 1:
                    try:
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    except:
                        avg_path_length = 0
                else:
                    avg_path_length = 0
            else:
                avg_path_length = 0
            
            degrees = [d for _, d in active_graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            
            try:
                cycles = list(nx.cycle_basis(active_graph))
                num_cycles = len(cycles)
                avg_cycle_length = np.mean([len(c) for c in cycles]) if cycles else 0
            except:
                num_cycles = 0
                avg_cycle_length = 0
            
            metrics = {
                'clustering': clustering,
                'avg_path_length': avg_path_length,
                'avg_degree': avg_degree,
                'num_edges': len(active_graph.edges),
                'num_nodes': len(active_graph.nodes),
                'num_cycles': num_cycles,
                'avg_cycle_length': avg_cycle_length if cycles else 0
            }
            
            self.network_metrics.append(metrics)
            return metrics
        else:
            return None
    
    def update(self):
        self._update_positions()
        
        self._update_predator()
        
        self._check_predation()
        
        self._check_predator_infection()
        
        self.update_infection_status()
        
        self.compute_network_metrics()
        
        self.eaten_history.append(self.eaten_count)
    
    def _update_positions(self):
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) == 0:
            return
            
        random_movement = np.zeros_like(self.velocities)
        random_movement[active_agents] = np.random.normal(0, 1.0, (len(active_agents), 2))
        
        random_movement[active_agents] *= self.max_speed
        
        structural_positions = self._calculate_tree_positions()
        
        for i in active_agents:
            direction_to_structure = structural_positions[i] - self.positions[i]
            
            if np.linalg.norm(direction_to_structure) > 0:
                direction_to_structure = direction_to_structure / np.linalg.norm(direction_to_structure) * self.max_speed * 0.5
                
            random_movement[i] += direction_to_structure * 0.1
            
            if self.infection_status[i] == 1:
                neighbors = list(self.tree.neighbors(i))
                active_uninfected_neighbors = [n for n in neighbors if self.active_status[n] == 1 and self.infection_status[n] == 0]
                
                if active_uninfected_neighbors:
                    target = random.choice(active_uninfected_neighbors)
                    direction_to_target = self.positions[target] - self.positions[i]
                    if np.linalg.norm(direction_to_target) > 0:
                        direction_to_target = direction_to_target / np.linalg.norm(direction_to_target) * self.max_speed * 2
                        random_movement[i] += direction_to_target
            
            predator_direction = self.positions[i] - self.predator_position
            distance_to_predator = np.linalg.norm(predator_direction)
            if distance_to_predator < self.infection_radius * 3:
                if np.linalg.norm(predator_direction) > 0:
                    predator_direction = predator_direction / np.linalg.norm(predator_direction) * self.max_speed
                    random_movement[i] += self.predator_avoidance_weight * predator_direction
            
            if self.alarm_counters[i] > 0:
                random_movement[i] *= self.alarm_acceleration
        
        self.positions[active_agents] += random_movement[active_agents]
        
        self.positions = np.clip(self.positions, 0, self.world_size)
        
        self.velocities = random_movement
    
    def _update_predator(self):
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) == 0:
            random_dir = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(random_dir) > 0:
                self.predator_velocity = random_dir / np.linalg.norm(random_dir) * self.predator_speed
            self.predator_position += self.predator_velocity
            self.predator_position = self.predator_position % self.world_size
            return
            
        distances = np.linalg.norm(self.positions[active_agents] - self.predator_position, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_agent_idx = active_agents[nearest_idx]
        
        direction = self.positions[nearest_agent_idx] - self.predator_position
        if np.linalg.norm(direction) > 0:
            self.predator_velocity = direction / np.linalg.norm(direction) * self.predator_speed
        
        random_direction = np.random.uniform(-1, 1, 2) * 0.3
        if np.linalg.norm(self.predator_velocity + random_direction) > 0:
            self.predator_velocity = (self.predator_velocity + random_direction)
            self.predator_velocity = self.predator_velocity / np.linalg.norm(self.predator_velocity) * self.predator_speed
        
        self.predator_position += self.predator_velocity
        
        self.predator_position = self.predator_position % self.world_size
    
    def _check_predation(self):
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) == 0:
            return
            
        distances = np.linalg.norm(self.positions[active_agents] - self.predator_position, axis=1)
        
        eaten_indices = [active_agents[i] for i, d in enumerate(distances) if d < self.predation_radius]
        
        if eaten_indices:
            self.active_status[eaten_indices] = 0
            self.eaten_count += len(eaten_indices)
            
            self.infection_status[eaten_indices] = 0
    
    def _check_predator_infection(self):
        predator_radius = self.infection_radius * 1.5
        
        active_agents = np.where(self.active_status == 1)[0]
        if len(active_agents) == 0:
            return
            
        distances = np.linalg.norm(self.positions[active_agents] - self.predator_position, axis=1)
        
        close_indices = [active_agents[i] for i, d in enumerate(distances) if d < predator_radius and d >= self.predation_radius]
        
        if close_indices:
            self.infection_status[close_indices] = 1
            
            self.alarm_counters[close_indices] = self.alarm_duration
    
    def get_statistics(self):
        active_agents = np.sum(self.active_status)
        active_infected = np.sum((self.infection_status == 1) & (self.active_status == 1))
        
        return {
            'num_active': active_agents,
            'num_infected': active_infected,
            'num_eaten': self.eaten_count,
            'infection_percentage': (active_infected / active_agents * 100) if active_agents > 0 else 0,
            'eaten_percentage': (self.eaten_count / self.num_agents * 100),
            'network_metrics': self.network_metrics[-1] if self.network_metrics else None
        }

    def visualize(self, ax=None):
        """Visualize the current state of the simulation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.clear()
        
        active_nodes = np.where(self.active_status == 1)[0]
        for edge in self.tree.edges():
            i, j = edge
            # Only draw edges between active nodes
            if i in active_nodes and j in active_nodes:
                ax.plot([self.positions[i, 0], self.positions[j, 0]], 
                        [self.positions[i, 1], self.positions[j, 1]], 
                        'k-', alpha=0.2, linewidth=0.5)
        
        # agents
        susceptible = np.where((self.infection_status == 0) & (self.active_status == 1))[0]
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        eaten = np.where(self.active_status == 0)[0]
        
        # Plot susceptible agents
        if len(susceptible) > 0:
            ax.scatter(
                self.positions[susceptible, 0], 
                self.positions[susceptible, 1], 
                color='blue', s=30, alpha=0.7, label='Susceptible'
            )
        
        # Plot alarmed agents
        if len(infected) > 0:
            ax.scatter(
                self.positions[infected, 0], 
                self.positions[infected, 1], 
                color='red', s=50, alpha=0.7, label='Alarmed'
            )
            
            # Add velocity vectors for alarmed agents
            ax.quiver(
                self.positions[infected, 0],
                self.positions[infected, 1],
                self.velocities[infected, 0],
                self.velocities[infected, 1],
                color='red', scale=20, width=0.005
            )
        
        # Plot eaten agents
        if len(eaten) > 0:
            ax.scatter(
                self.positions[eaten, 0], 
                self.positions[eaten, 1], 
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
            self.infection_radius * 1.5, 
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
        
        # Set plot limits
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        # ax.set_title(f'Tree-Like Network Model (Active: {np.sum(self.active_status)}, Alarmed: {np.sum((self.infection_status == 1) & (self.active_status == 1))}, Eaten: {self.eaten_count})')
        ax.set_title(f'Tree-Like Fish School Network (Active: {np.sum(self.active_status)}, Alarmed: {np.sum((self.infection_status == 1) & (self.active_status == 1))}, Eaten: {self.eaten_count})')
        

        # Add legend
        ax.legend(loc='upper right')
        
        return ax

def run_simulation(model, num_steps=100, save_animation=False, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    active_agents = np.where(model.active_status == 1)[0]
    distances = np.linalg.norm(model.positions[active_agents] - model.predator_position, axis=1)
    closest_idx = active_agents[np.argmin(distances)]
    model.infection_status[closest_idx] = 1
    model.alarm_counters[closest_idx] = model.alarm_duration
    
    def update(frame):
        model.update()
        ax1 = model.visualize(ax)
        
        stats = model.get_statistics()
        stats_text = f"Step: {frame}\n"
        stats_text += f"Active Agents: {stats['num_active']}\n"
        stats_text += f"Alarmed: {stats['num_infected']} ({stats['infection_percentage']:.1f}%)\n"
        stats_text += f"Eaten: {stats['num_eaten']} ({stats['eaten_percentage']:.1f}%)"
        
        if stats['network_metrics']:
            stats_text += f"\nAvg Degree: {stats['network_metrics']['avg_degree']:.2f}\n"
            stats_text += f"Clustering: {stats['network_metrics']['clustering']:.2f}\n"
            stats_text += f"Closed Loops: {stats['network_metrics']['num_cycles']}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax1
    
    ani = FuncAnimation(fig, update, frames=range(num_steps), interval=100, blit=False)
    
    if save_animation:
        if filename is None:
            filename = 'tree_network_simulation.gif'
        ani.save(filename, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def analyze_network(model):
    if len(model.graph.edges) == 0:
        print("No interactions recorded yet.")
        return
    
    plt.figure(figsize=(12, 10))
    
    pos = {i: tuple(model.positions[i]) for i in model.graph.nodes()}
    
    node_colors = ['red' if model.infection_status[i] == 1 else 'blue' for i in model.graph.nodes()]
    
    node_sizes = [100 if model.active_status[i] == 1 else 50 for i in model.graph.nodes()]
    node_alphas = [0.7 if model.active_status[i] == 1 else 0.3 for i in model.graph.nodes()]
    
    nx.draw_networkx(
        model.graph, 
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        # alpha=node_alphas,
        with_labels=True,
        font_size=8
    )
    
    plt.title('Tree-Like Network Structure')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\nNetwork Metrics:")
    metrics = model.compute_network_metrics()
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Check for closed loops (cycles)
        print(f"\nNumber of cycles (closed loops): {metrics['num_cycles']}")
        
        # Calculate average cycle length
        if metrics['num_cycles'] > 0:
            print(f"Average cycle length: {metrics['avg_cycle_length']:.2f}")
        
        # Compare to random graph to check for small-world properties
        if metrics['avg_path_length'] > 0:
            # Generate random graph with same number of nodes and edges
            random_graph = nx.gnm_random_graph(metrics['num_nodes'], metrics['num_edges'])
            
            # Calculate clustering and path length for random graph
            try:
                random_clustering = nx.average_clustering(random_graph)
                
                # Calculate average path length for random graph
                random_path_length = 0
                try:
                    random_path_length = nx.average_shortest_path_length(random_graph)
                except nx.NetworkXError:
                    # Graph is not connected
                    pass
                
                if random_path_length > 0:
                    # Small world index
                    sigma = (metrics['clustering'] / random_clustering) / (metrics['avg_path_length'] / random_path_length)
                    print(f"Small-world index (σ): {sigma:.4f}")
                    print(f"  (σ > 1 indicates small-world properties)")
            except:
                pass
    else:
        print("No metrics available")

# Function to analyze information propagation over time
def analyze_information_propagation(model, num_steps=100):
    """Run the simulation and analyze information propagation over time."""
    # Reset the model
    model.reset()
    
    # Start with one infected agent near the predator
    active_agents = np.where(model.active_status == 1)[0]
    distances = np.linalg.norm(model.positions[active_agents] - model.predator_position, axis=1)
    closest_idx = active_agents[np.argmin(distances)]
    model.infection_status[closest_idx] = 1
    model.alarm_counters[closest_idx] = model.alarm_duration
    
    # Run the simulation for specified steps
    infection_history = []
    network_metrics_history = []
    cycles_history = []
    
    for step in range(num_steps):
        model.update()
        
        # Record infection count
        active_infected = np.sum((model.infection_status == 1) & (model.active_status == 1))
        infection_history.append(active_infected)
        
        # Compute and store network metrics
        metrics = model.compute_network_metrics()
        if metrics:
            network_metrics_history.append(metrics)
            cycles_history.append(metrics['num_cycles'])
    
    # Plot infection propagation over time
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Infection propagation
    plt.subplot(3, 1, 1)
    plt.plot(infection_history, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Alarmed Agents')
    plt.title('Alarm Signal Propagation in Tree-Like Network')
    plt.grid(True)
    
    # Plot 2: Network metrics over time
    plt.subplot(3, 1, 2)
    if network_metrics_history:
        plt.plot([m['avg_degree'] for m in network_metrics_history], 'b-', label='Avg Degree')
        plt.plot([m['clustering'] for m in network_metrics_history], 'g-', label='Clustering')
        if any(m['avg_path_length'] > 0 for m in network_metrics_history):
            plt.plot([m['avg_path_length'] if m['avg_path_length'] > 0 else None 
                    for m in network_metrics_history], 'y-', label='Avg Path Length')
    
    plt.xlabel('Time Step')
    plt.ylabel('Metric Value')
    plt.title('Network Metrics Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Number of cycles (closed loops) over time
    plt.subplot(3, 1, 3)
    if cycles_history:
        plt.plot(cycles_history, 'm-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Number of Closed Loops')
        plt.title('Closed Loops in Tree-Like Network Over Time')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return the results
    return {
        'infection_history': infection_history,
        'network_metrics': network_metrics_history,
        'cycles_history': cycles_history
    }

# Example usage
if __name__ == "__main__":
    # Create model with parameters
    model = TreeNetworkSISModel(
        num_agents=50,
        world_size=100,
        infection_radius=7,
        recovery_rate=0.05,
        infection_rate=0.8,
        max_speed=1.0,
        predator_speed=2.0,
        tree_depth=3,
        branching_factor=3,
        position_jitter=5.0,
        alarm_acceleration=2.0,      # Agent speed multiplier during alarm
        alarm_duration=10,           # How long the alarm acceleration lasts
        predation_radius=3.0         # Radius within which agents are eaten
    )
    
    # Run the simulation
    animation = run_simulation(model, num_steps=200, save_animation=True, filename='tree_network_simulation.gif')
    
    # Analyze the final network structure
    analyze_network(model)
    
    # Analyze information propagation
    results = analyze_information_propagation(model, num_steps=100)
