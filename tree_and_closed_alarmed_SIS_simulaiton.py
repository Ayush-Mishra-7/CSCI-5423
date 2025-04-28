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
                 separation_weight=0.4, random_weight=0.1, predator_avoidance_weight=2.0,
                 alarm_acceleration=2.5, alarm_duration=10, predation_radius=3.0):
     
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
        
        self.alarm_acceleration = alarm_acceleration
        self.alarm_duration = alarm_duration
        self.predation_radius = predation_radius
        
        self.fish_positions = np.random.uniform(0, world_size, (num_fish, 2))
        self.fish_velocities = np.random.uniform(-1, 1, (num_fish, 2))
        self.fish_velocities = self.fish_velocities / np.linalg.norm(self.fish_velocities, axis=1)[:, np.newaxis] * fish_speed
        
        self.predator_position = np.array([world_size * 0.8, world_size * 0.8])
        pred_vel = np.random.uniform(-1, 1, 2)
        self.predator_velocity = pred_vel / np.linalg.norm(pred_vel) * predator_speed
        
        self.infection_status = np.zeros(num_fish)
        
        self.alarm_counters = np.zeros(num_fish)
        
        self.active_status = np.ones(num_fish)
        
        self.eaten_fish_count = 0
        
        self.graph = nx.Graph()
        self.interaction_history = []
        
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
        
    def reset(self):
        self.infection_status = np.zeros(self.num_fish)
        self.alarm_counters = np.zeros(self.num_fish)
        self.active_status = np.ones(self.num_fish)
        self.eaten_fish_count = 0
        self.interaction_history = []
        self.infection_history = []
        self.network_metrics = []
        self.eaten_history = []
        
    def initialize_infection(self, num_initial=1):
        active_fish = np.where(self.active_status == 1)[0]
        if len(active_fish) > 0:
            num_to_infect = min(num_initial, len(active_fish))
            initial_infected = np.random.choice(active_fish, num_to_infect, replace=False)
            self.infection_status[initial_infected] = 1
        
    def update_infection_status(self):
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        
        active_fish = np.where(self.active_status == 1)[0]
        
        if len(active_fish) < 2:  # Need at least 2 fish for distance calculation
            return
        
        active_positions = self.fish_positions[active_fish]
        distances = distance.cdist(active_positions, active_positions)
        
        susceptible = np.where((self.infection_status == 0) & (self.active_status == 1))[0]
        newly_infected = []
        
        active_to_orig = {i: active_fish[i] for i in range(len(active_fish))}
        orig_to_active = {fish_idx: i for i, fish_idx in enumerate(active_fish)}
        
        for s_idx, s in enumerate(susceptible):
            s_active_idx = orig_to_active.get(s, None)
            if s_active_idx is None:
                continue
                
            for i in infected:
                i_active_idx = orig_to_active.get(i, None)
                if i_active_idx is None:
                    continue
                    
                if distances[s_active_idx, i_active_idx] <= self.infection_radius:
                    if np.random.random() < self.infection_rate:
                        newly_infected.append(s)
                        self.graph.add_edge(s, i)
                        
                        self.alarm_counters[s] = self.alarm_duration
                        break
        
        self.infection_status[newly_infected] = 1
        
        self.alarm_counters = np.maximum(0, self.alarm_counters - 1)
        
        for i in infected:
            if np.random.random() < self.recovery_rate:
                self.infection_status[i] = 0
        
        active_infected = np.sum((self.infection_status == 1) & (self.active_status == 1))
        self.infection_history.append(active_infected)
        
    def compute_network_metrics(self):
        active_fish = np.where(self.active_status == 1)[0]
        if len(active_fish) == 0:
            return None
            
        active_graph = self.graph.subgraph(active_fish)
        
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
            
            metrics = {
                'clustering': clustering,
                'avg_path_length': avg_path_length,
                'avg_degree': avg_degree,
                'num_edges': len(active_graph.edges),
                'num_nodes': len(active_graph.nodes)
            }
            
            self.network_metrics.append(metrics)
            return metrics
        else:
            return None
        
    def update(self):
        self._update_fish_velocities()
        
        active_fish = np.where(self.active_status == 1)[0]
        self.fish_positions[active_fish] += self.fish_velocities[active_fish]
        
        self.fish_positions = self.fish_positions % self.world_size
        
        self._update_predator()
        
        self._check_predation()
        
        self._check_predator_infection()
        
        self.update_infection_status()
        
        self.compute_network_metrics()
        
        self.eaten_history.append(self.eaten_fish_count)
        
    def _update_fish_velocities(self):
        new_velocities = np.zeros_like(self.fish_velocities)
        
        active_fish = np.where(self.active_status == 1)[0]
        
        for i in active_fish:
            distances = np.linalg.norm(self.fish_positions[active_fish] - self.fish_positions[i], axis=1)
            neighbors_idx = np.where(distances < self.infection_radius * 2)[0]
            neighbors = [active_fish[j] for j in neighbors_idx if active_fish[j] != i]
            
            if len(neighbors) > 0:
                alignment = np.mean(self.fish_velocities[neighbors], axis=0)
                if np.linalg.norm(alignment) > 0:
                    alignment = alignment / np.linalg.norm(alignment) * self.fish_speed
                
                cohesion = np.mean(self.fish_positions[neighbors], axis=0) - self.fish_positions[i]
                if np.linalg.norm(cohesion) > 0:
                    cohesion = cohesion / np.linalg.norm(cohesion) * self.fish_speed
                
                separation = np.zeros(2)
                close_neighbors = [n for n in neighbors if np.linalg.norm(self.fish_positions[i] - self.fish_positions[n]) < self.infection_radius * 0.5]
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
                        
                        if distance_to_predator < self.infection_radius * 1.5 and self.alarm_counters[i] == 0:
                            self.alarm_counters[i] = self.alarm_duration
                            if self.infection_status[i] == 0:
                                self.infection_status[i] = 1
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
                    
                    speed_multiplier = 1.0
                    
                    if self.infection_status[i] == 1:
                        speed_multiplier *= 1.5
                        
                    if self.alarm_counters[i] > 0:
                        speed_multiplier *= self.alarm_acceleration
                        
                    velocity *= speed_multiplier
                
                new_velocities[i] = velocity
            else:
                new_velocities[i] = self.fish_velocities[i]
        
        self.fish_velocities = new_velocities
    
    def _update_predator(self):
        active_fish = np.where(self.active_status == 1)[0]
        if len(active_fish) == 0:
            random_dir = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(random_dir) > 0:
                self.predator_velocity = random_dir / np.linalg.norm(random_dir) * self.predator_speed
            self.predator_position += self.predator_velocity
            self.predator_position = self.predator_position % self.world_size
            return
            
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_fish_idx = active_fish[nearest_idx]
        
        direction = self.fish_positions[nearest_fish_idx] - self.predator_position
        if np.linalg.norm(direction) > 0:
            self.predator_velocity = direction / np.linalg.norm(direction) * self.predator_speed
        
        self.predator_position += self.predator_velocity
        
        self.predator_position = self.predator_position % self.world_size
    
    def _check_predation(self):
        active_fish = np.where(self.active_status == 1)[0]
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        
        eaten_indices = [active_fish[i] for i, d in enumerate(distances) if d < self.predation_radius]
        
        if eaten_indices:
            self.active_status[eaten_indices] = 0
            self.eaten_fish_count += len(eaten_indices)
            
            self.infection_status[eaten_indices] = 0
            
            print(f"Predator ate {len(eaten_indices)} fish! Total eaten: {self.eaten_fish_count}")
    
    def _check_predator_infection(self):
        predator_radius = self.infection_radius * 1.5
        
        active_fish = np.where(self.active_status == 1)[0]
        distances = np.linalg.norm(self.fish_positions[active_fish] - self.predator_position, axis=1)
        
        close_indices = [active_fish[i] for i, d in enumerate(distances) if d < predator_radius and d >= self.predation_radius]
        
        if close_indices:
            self.infection_status[close_indices] = 1
            
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
            'network_metrics': self.network_metrics[-1] if self.network_metrics else None
        }

    def visualize(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.clear()
        
        susceptible = np.where((self.infection_status == 0) & (self.active_status == 1))[0]
        infected = np.where((self.infection_status == 1) & (self.active_status == 1))[0]
        
        eaten = np.where(self.active_status == 0)[0]
        
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
                color='red', s=50, alpha=0.7, label='Alarmed'
            )
            
            ax.quiver(
                self.fish_positions[infected, 0],
                self.fish_positions[infected, 1],
                self.fish_velocities[infected, 0],
                self.fish_velocities[infected, 1],
                color='red', scale=20, width=0.005
            )
        
        if len(eaten) > 0:
            ax.scatter(
                self.fish_positions[eaten, 0], 
                self.fish_positions[eaten, 1], 
                color='gray', s=20, alpha=0.3, label='Eaten'
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
        
        predation_circle = plt.Circle(
            (self.predator_position[0], self.predator_position[1]), 
            self.predation_radius, 
            color='red', fill=False, linestyle='--', alpha=0.5
        )
        ax.add_patch(predation_circle)
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Fish School SIS Model (Active: {np.sum(self.active_status)}, Alarmed: {np.sum((self.infection_status == 1) & (self.active_status == 1))}, Eaten: {self.eaten_fish_count})')
        
        ax.legend(loc='upper right')
        
        return ax

def run_simulation(model, num_steps=100, save_animation=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    active_fish = np.where(model.active_status == 1)[0]
    distances = np.linalg.norm(model.fish_positions[active_fish] - model.predator_position, axis=1)
    closest_idx = active_fish[np.argmin(distances)]
    model.infection_status[closest_idx] = 1
    
    def update(frame):
        model.update()
        ax1 = model.visualize(ax)
        
        stats = model.get_statistics()
        stats_text = f"Step: {frame}\n"
        stats_text += f"Active Fish: {stats['num_active']}\n"
        stats_text += f"Alarmed: {stats['num_infected']} ({stats['infection_percentage']:.1f}%)\n"
        stats_text += f"Eaten: {stats['num_eaten']} ({stats['eaten_percentage']:.1f}%)"
        
        if stats['network_metrics']:
            stats_text += f"\nAvg Degree: {stats['network_metrics']['avg_degree']:.2f}\n"
            stats_text += f"Clustering: {stats['network_metrics']['clustering']:.2f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax1
    
    ani = FuncAnimation(fig, update, frames=range(num_steps), interval=100, blit=False)
    
    if save_animation:
        ani.save('fish_school_predation_model_new.gif', writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

if __name__ == "__main__":
    model = FishSchoolSISModel(
        num_fish=50,
        world_size=100,
        infection_radius=7,
        recovery_rate=0.05,
        infection_rate=0.8,
        fish_speed=1.5,
        predator_speed=2.5,
        alarm_acceleration=2.5,      # Fish speed multiplier during alarm
        alarm_duration=10,           # How long the alarm acceleration lasts
        predation_radius=3.0         # Radius within which fish are eaten
    )
    
    animation = run_simulation(model, num_steps=200, save_animation=True)
