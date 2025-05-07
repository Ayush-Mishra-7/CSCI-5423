# Network Structure Generation and SIS Simulation

## Overview

The `merged_code.py` script is the final updated version that integrates network structure generation capabilities with SIS epidemic simulation. It allows users to:

1. Generate various types of network topologies
2. Run SIS simulations on these generated networks
3. Analyze and visualize the spread of information across different network structures

## Requirements

To run this code, you need the following Python packages:

```
numpy
matplotlib
networkx
scipy
pandas
```

## How to Run

To run the network generation and SIS simulation, execute the `merged_code.py` file:

```
python merged_code.py
```

### Configuration

The script contains various parameters that can be modified directly in the code:

- Network structure generation parameters:
  - Network type
  - Number of nodes
  - Connection probability/average degree
  - Rewiring probability
  - Fish, Predator Speed  (To replicate real-life scenarios)
  - Alarm Accelaration (Bump in fish speed when information of predator is known)

- SIS epidemic simulation parameters:
  - Beta (infection rate): Probability of infection(alarmed) spreading from an infected node to a susceptible(unalarmed) neighbor
  - Gamma (recovery rate): Probability of an infected(alarmed) node recovering and becoming susceptible(unalarmed) again
  - Initial infection percentage: Percentage of nodes initially infected
  - Number of simulation steps/time periods
  - Simulation iterations: Number of times to run the simulation to average results

The parameters can be adjusted by editing their values in the script. Key parameters are typically defined at the top of the script or within clearly marked configuration sections.

## Outputs

When you run `merged_code.py`, you'll get the following outputs:

1. **Network Structure Visualization**: 
   - Visual representation of the generated network topology
   - Network connections (edges) showing the potential transmission paths

2. **SIS Progression**:
   - Real-time or step-by-step visualization of infection spread through the network
   - Time-series plots showing:
     - Number/percentage of infected(alarmed) nodes over time
     - Number/percentage of susceptible(unalarmed) nodes over time

3. **Network Analysis Metrics**:
   - Structural properties of the generated network:
     - Average degree and degree distribution
     - Clustering coefficient
     - Average path length
   - How these properties influence epidemic dynamics

4. **Simulation Statistics**:
   - Terminal output or saved results

## Code Structure

The `merged_code.py` script is organized into several key sections:

1. **Imports and Configuration**: Library imports and parameter settings
2. **Network Generation Functions**: Code for creating different network structures
3. **SIS Model Implementation**: The simulation logic
4. **Visualization Functions**: Code for plotting and visual analysis
5. **Main Execution**: The primary workflow that ties everything together

## Data Sources for Future work on real dataset

### Fish
- **Zenodo Dataset**: Raw videos of empirical observations studying large shoals of sulphur mollies (P. sulphuraria). Data includes cluster metrics such as area, volume, time duration, and speed.
  - [Zenodo Record](https://zenodo.org/records/7323527)
  
- **GitHub Repository**: Code for deep learning-based analysis using LSTM models to detect perturbations in cellular automata models simulating fish shoals.
  - [GitHub Repository](https://github.com/RobertTLange/automata-perturbation-lstm)

- **Figshare Supplementary Files**: Supplementary data for the study on defensive shimmering responses in *Apis dorsata*, triggered by dark stimuli moving against a bright background.
  - [Figshare Record](https://figshare.com/articles/journal_contribution/Supplementary_files_for_the_article_Defensive_shimmering_responses_in_Apis_dorsata_are_triggered_by_dark_stimuli_moving_against_a_bright_background_/20359875/3)

### Shared Dataset and Starter Code
These two resources are linked to the same study:
- **Starter Code**: Includes tools for analyzing the dataset.
  - [Zenodo Record](https://zenodo.org/records/4983257)
- **Database**: Contains the raw data for analysis.
  - [Dryad Database](https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6x5x)


