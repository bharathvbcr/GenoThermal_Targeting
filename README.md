# Geno-Thermal Targeting

**Patient-Specific Magnetic Nanoparticle Therapy Design System**

## Project Overview

This project implements an end-to-end computational pipeline for designing **patient-specific magnetic nanoparticle therapy systems**. The system identifies cancer-specific surface markers through genomic analysis, engineers custom peptide ligands to bind them, and validates the complete assembly through physics simulation.

Unlike standard targeted therapy design which relies on passive prediction, this project employs **generative inverse design** to create molecular systems that don't exist in nature.

## "Hard Mode" Features

This project integrates advanced computational biology and AI techniques:

1. **Reinforcement Learning (PPO)**: Writes DNA sequences from scratch.
2. **Evolutionary Algorithms (GA)**: Optimizes multi-objective fitness functions across competing biological constraints.
3. **Computational Physics (Molecular Dynamics)**: Validates designs *in silico* using OpenMM.
4. **Biological Logic Gates**: Ensures kill switch activation ONLY under dual conditions (cancer context + hyperthermia).

## Pipeline Architecture

The workflow consists of 9 distinct phases:

1. **Phase 1: Genomic Discovery** (AlphaGenome) → Identify cancer-driving mutations (Super-Enhancers).
2. **Phase 2: Ligand Engineering** (AlphaFold) → Design peptides that specifically bind the target receptor.
3. **Phase 3: Structure Analysis & Visualization** → Score and visualize docking results in 3D.
4. **Phase 4: Evolutionary Promoter Design** (GA) → Evolve synthetic DNA promoters via genetic algorithms.
5. **Phase 5: Thermo-Switch Protein Design** → Engineer a protein that unfolds at 43°C.
6. **Phase 6: Nanoparticle Surface Topology** (MC) → Simulate ligand self-assembly on nanoparticle surfaces.
7. **Phase 7: Biological Circuit Integration** → Combine promoter + switch into an AND logic gate.
8. **Phase 8: RL-Driven Sequence Design** (PPO) → Train an AI agent to write DNA sequences.
9. **Phase 9: Physics Verification** (OpenMM/CUDA) → Molecular Dynamics simulations to prove thermal switching.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for OpenMM and AlphaFold tasks)

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/bharathvbcr/GenoThermal_Targeting.git
    cd Geno-Thermal_Targeting
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: You may need to install `alphagenome` manually if it's a private package or not on PyPI.*

3. Install the package in editable mode:

    ```bash
    pip install -e .
    ```

## Usage

### Master Pipeline

The entire workflow is orchestrated in the master Jupyter Notebook. This is the best place to start to understand the complete system.

```bash
jupyter notebook Geno_Thermal_Master.ipynb
```

### Standalone Scripts

You can also run individual components of the pipeline:

- **Genomic Discovery**:

    ```bash
    python genomic_discovery.py
    ```

- **Ligand Designer**:

    ```bash
    python ligand_designer.py
    ```

## Project Structure

- `Geno_Thermal_Master.ipynb`: The main notebook acting as the project controller and tutorial.
- `hard_mode/`: Core library code containing the implementations for GA, RL, and MD simulations.
- `genomic_discovery.py`: Script for Phase 1 analysis.
- `ligand_designer.py`: Script for Phase 2 peptide design.
- `requirements.txt`: Python package dependencies.
- `setup.py`: Package installation configuration.

## Dependencies

Key libraries used in this project include:

- **Bioinformatics**: `biopython`, `alphagenome`
- **AI/ML**: `gymnasium`, `stable-baselines3`, `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`, `py3Dmol`, `logomaker`
- **Infrastructure**: `jupyter`, `requests`
