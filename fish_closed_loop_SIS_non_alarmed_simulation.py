import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import random
from scipy.spatial import distance

class FishSchoolSISModel:
    def __init__(self, num_fish=100, world_size=100, infection_radius=10, 
                 recovery_rate=0.1, infection_rate=0.8, fish_speed=2.0, 
                 predator_speed=2.5, alignment_weight=0.3, cohesion_weight=0.3, 
                 separation_weight=0.4, random_weight=0.1, predator_avoidance_weight=2.0):
     
        self.num_fish = num_fish
        self.world_size = world_size
        self.infection_radius = infection_radius
        self.recovery_rate = recovery_rate
        self.infection_rate = infection_rate
        self.fish_speed = fish_speed
        self.predator_speed = predator_speed
        
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight
        self.random_weight = random_weight
        self.predator_avoidance_weight = predator_avoidance_weight
        
        self.fish_positions = np.random.uniform(0, world_size, (num_fish, 2))
        self.fish_velocities = np.random.uniform(-1, 1, (num_fish, 2))
        self.fish_velocities = self.fish_velocities / np.linalg.norm(self.fish_velocities, axis=1)[:, np.newaxis] * fish_speed
        
        self.predator_position = np.array([world_size * 0.8, world_size * 0.8])
        pred_vel = np.random.uniform(-1, 1, 2)
        self.predator_velocity = pred_vel / np.linalg.norm(pred_vel) * predator_speed
        
        self.infection_status = np.zeros(num_fish)
        
        self.graph = nx.Graph()
        self.interaction_history = []
        
        self.infection_history = []
        self.network_metrics = []
        
    def reset(self):
        self.infection_status = np.zeros(self.num_fish)
        self.interaction_history = []
        self.infection_history = []
        self.network_metrics = []
        
    def initialize_infection(self, num_initial=1):
        initial_infected = np.random.choice(self.num_fish, num_initial, replace=False)
        self.infection_status[initial_infected] = 1
        
    def update_infection_status(self):
        infected = np.where(self.infection_status == 1)[0]
        
        distances = distance.cdist(self.fish_positions, self.fish_positions)
        
        susceptible = np.where(self.infection_status == 0)[0]
        newly_infected = []
        
        for s in susceptible:
            close_infected = [i for i in infected if distances[s, i] <= self.infection_radius]
            if close_infected and np.random.random() < self.infection_rate:
                newly_infected.append(s)
                for i in close_infected:
                    self.graph.add_edge(s, i)
        
        self.infection_status[newly_infected] = 1
        
        for i in infected:
            if np.random.random() < self.recovery_rate:
                self.infection_status[i] = 0
        
        self.infection_history.append(np.sum(self.infection_status))
        
    def compute_network_metrics(self):
        if len(self.graph.edges) > 0:
            clustering = nx.average_clustering(self.graph)
            
            connected_components = list(nx.connected_components(self.graph))
            if connected_components:
                largest_component = max(connected_components, key=len)
                subgraph = self.graph.subgraph(largest_component)
                if len(subgraph) > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 0
            else:
                avg_path_length = 0
            
            degrees = [d for _, d in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            
            metrics = {
                'clustering': clustering,
                'avg_path_length': avg_path_length,
                'avg_degree': avg_degree,
                'num_edges': len(self.graph.edges),
                'num_nodes': len(self.graph.nodes)
            }
            
            self.network_metrics.append(metrics)
            return metrics
        else:
            return None
        
    def update(self):
        self._update_fish_velocities()
        self.fish_positions += self.fish_velocities
        
        self.fish_positions = self.fish_positions % self.world_size
        
        self._update_predator()
        
        self._check_predator_infection()
        
        self.update_infection_status()
        
        self.compute_network_metrics()
        
    def _update_fish_velocities(self):
        new_velocities = np.zeros_like(self.fish_velocities)
        
        for i in range(self.num_fish):
            distances = np.linalg.norm(self.fish_positions - self.fish_positions[i], axis=1)
            neighbors = np.where((distances > 0) & (distances < self.infection_radius * 2))[0]
            
            if len(neighbors) > 0:
                alignment = np.mean(self.fish_velocities[neighbors], axis=0)
                if np.linalg.norm(alignment) > 0:
                    alignment = alignment / np.linalg.norm(alignment) * self.fish_speed
                
                cohesion = np.mean(self.fish_positions[neighbors], axis=0) - self.fish_positions[i]
                if np.linalg.norm(cohesion) > 0:
                    cohesion = cohesion / np.linalg.norm(cohesion) * self.fish_speed
                
                separation = np.zeros(2)
                close_neighbors = np.where((distances > 0) & (distances < self.infection_radius * 0.5))[0]
                if len(close_neighbors) > 0:
                    for neighbor in close_neighbors:
                        diff = self.fish_positions[i] - self.fish_positions[neighbor]
                        if np.linalg.norm(diff) > 0:
                            separation += diff / (np.linalg.norm(diff) ** 2)
                    if np.linalg.norm(separation) > 0:
                        separation = separation / np.linalg.norm(separation) * self.fish_speed
                
                random_direction = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(random_direction) > 0:
                    random_direction = random_direction / np.linalg.norm(random_direction) * self.fish_speed
                
                predator_direction = self.fish_positions[i] - self.predator_position
                distance_to_predator = np.linalg.norm(predator_direction)
                if distance_to_predator < self.infection_radius * 3:
                    if np.linalg.norm(predator_direction) > 0:
                        predator_direction = predator_direction / np.linalg.norm(predator_direction) * self.fish_speed
                else:
                    predator_direction = np.zeros(2)
                
                velocity = (
                    self.alignment_weight * alignment +
                    self.cohesion_weight * cohesion +
                    self.separation_weight * separation +
                    self.random_weight * random_direction +
                    self.predator_avoidance_weight * predator_direction
                )
                
                if np.linalg.norm(velocity) > 0:
                    velocity = velocity / np.linalg.norm(velocity) * self.fish_speed
                    
                    if self.infection_status[i] == 1:
                        velocity *= 1.5
                
                new_velocities[i] = velocity
            else:
                new_velocities[i] = self.fish_velocities[i]
        
        self.fish_velocities = new_velocities
    
    def _update_predator(self):
        distances = np.linalg.norm(self.fish_positions - self.predator_position, axis=1)
        nearest_fish_idx = np.argmin(distances)
        
        direction = self.fish_positions[nearest_fish_idx] - self.predator_position
        if np.linalg.norm(direction) > 0:
            self.predator_velocity = direction / np.linalg.norm(direction) * self.predator_speed
        
        self.predator_position += self.predator_velocity
        
        self.predator_position = self.predator_position % self.world_size
    
    def _check_predator_infection(self):
        predator_radius = self.infection_radius * 1.5
        distances = np.linalg.norm(self.fish_positions - self.predator_position, axis=1)
        
        close_to_predator = np.where(distances < predator_radius)[0]
        self.infection_status[close_to_predator] = 1
    
    def get_statistics(self):
        return {
            'num_infected': np.sum(self.infection_status),
            'infection_percentage': np.sum(self.infection_status) / self.num_fish * 100,
            'network_metrics': self.network_metrics[-1] if self.network_metrics else None
        }

    def visualize(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.clear()
        
        susceptible = np.where(self.infection_status == 0)[0]
        infected = np.where(self.infection_status == 1)[0]
        
        if len(susceptible) > 0:
            ax.scatter(
                self.fish_positions[susceptible, 0], 
                self.fish_positions[susceptible, 1], 
                color='blue', s=30, alpha=0.7, label='Susceptible'
            )
        
        if len(infected) > 0:
            ax.scatter(
                self.fish_positions[infected, 0], 
                self.fish_positions[infected, 1], 
                color='red', s=50, alpha=0.7, label='Infected (Alarmed)'
            )
        
        ax.scatter(
            self.predator_position[0], 
            self.predator_position[1], 
            color='black', s=200, marker='*', label='Predator'
        )
        
        predator_circle = plt.Circle(
            (self.predator_position[0], self.predator_position[1]), 
            self.infection_radius * 1.5, 
            color='black', fill=False, linestyle='--', alpha=0.3
        )
        ax.add_patch(predator_circle)
        
        if len(infected) > 0:
            ax.quiver(
                self.fish_positions[infected, 0],
                self.fish_positions[infected, 1],
                self.fish_velocities[infected, 0],
                self.fish_velocities[infected, 1],
                color='red', scale=20, width=0.005
            )
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Fish School SIS Model (Infected: {np.sum(self.infection_status)})')
        
        ax.legend(loc='upper right')
        
        return ax

def run_simulation(model, num_steps=100, save_animation=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    distances = np.linalg.norm(model.fish_positions - model.predator_position, axis=1)
    closest_idx = np.argmin(distances)
    model.infection_status[closest_idx] = 1
    
    def update(frame):
        model.update()
        ax1 = model.visualize(ax)
        
        stats = model.get_statistics()
        stats_text = f"Step: {frame}\nInfected: {stats['num_infected']} ({stats['infection_percentage']:.1f}%)"
        if stats['network_metrics']:
            stats_text += f"\nAvg Degree: {stats['network_metrics']['avg_degree']:.2f}"
            stats_text += f"\nClustering: {stats['network_metrics']['clustering']:.2f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax1
    
    ani = FuncAnimation(fig, update, frames=range(num_steps), interval=100, blit=False)
    
    if save_animation:
        ani.save('fish_school_sis_model_extreme_alarm_predator_2x.gif', writer='ffmpeg', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def analyze_network(model):
    if len(model.graph.edges) == 0:
        print("No interactions recorded yet.")
        return
    
    plt.figure(figsize=(12, 10))
    
    pos = {i: tuple(model.fish_positions[i]) for i in model.graph.nodes()}
    
    node_colors = ['red' if model.infection_status[i] == 1 else 'blue' for i in model.graph.nodes()]
    
    nx.draw_networkx(
        model.graph, 
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        node_size=100,
        font_size=8,
        alpha=0.7
    )
    
    plt.title('Fish School Interaction Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print network metrics
    print("\nNetwork Metrics:")
    metrics = model.compute_network_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Check for closed loops (cycles)
    cycles = list(nx.cycle_basis(model.graph))
    print(f"\nNumber of cycles (closed loops): {len(cycles)}")
    
    # Calculate average cycle length
    if cycles:
        avg_cycle_len = np.mean([len(cycle) for cycle in cycles])
        print(f"Average cycle length: {avg_cycle_len:.2f}")
    
    if len(model.graph.edges) > 0 and metrics['avg_path_length'] > 0:
        random_graph = nx.gnm_random_graph(len(model.graph.nodes), len(model.graph.edges))
        
        random_clustering = nx.average_clustering(random_graph)
        
        try:
            random_path_length = nx.average_shortest_path_length(random_graph)
        except nx.NetworkXError:
            random_path_length = 0
        
        if random_path_length > 0:
            # Small world index
            sigma = (metrics['clustering'] / random_clustering) / (metrics['avg_path_length'] / random_path_length)
            print(f"Small-world index (σ): {sigma:.4f}")
            print(f"  (σ > 1 indicates small-world properties)")

def analyze_information_propagation(model, num_steps=100):
    model.reset()
    
    distances = np.linalg.norm(model.fish_positions - model.predator_position, axis=1)
    closest_idx = np.argmin(distances)
    model.infection_status[closest_idx] = 1
    
    # Run the simulation for specified steps
    infection_history = []
    network_metrics_history = []
    
    for step in range(num_steps):
        model.update()
        infection_history.append(np.sum(model.infection_status))
        
        # Compute and store network metrics
        metrics = model.compute_network_metrics()
        if metrics:
            network_metrics_history.append(metrics)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(infection_history, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Infected Fish')
    plt.title('Information Propagation in Fish School')
    plt.grid(True)
    
    ##TODO: fix
    plt.subplot(2, 1, 2)
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
    
    plt.tight_layout()
    plt.show()
    
    # Return the results
    return {
        'infection_history': infection_history,
        'network_metrics': network_metrics_history
    }

if __name__ == "__main__":
    model = FishSchoolSISModel(
        num_fish=20,
        world_size=100,
        infection_radius=5,
        recovery_rate=0.1,
        infection_rate=0.8,
        fish_speed=2.0,
        predator_speed=4.0
    )
    
    animation = run_simulation(model, num_steps=200, save_animation=True)
    
    analyze_network(model)
    
    results = analyze_information_propagation(model, num_steps=100)
