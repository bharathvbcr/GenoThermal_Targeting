# Geno-Thermal Targeting: Patient-Specific Magnetic Nanoparticle Therapy

Geno-Thermal Targeting is an end-to-end computational pipeline for the **generative design and verification** of personalized cancer therapies. The system combines genomic analysis, protein engineering, and biological circuit logic to create a nanoparticle-based treatment that activates exclusively under the dual conditions of **cancer-specific biomarkers** and **localized hyperthermia (43°C)**.

---

## 🔬 Project Architecture: The 10-Phase Pipeline

The workflow is divided into ten integrated phases, moving from raw patient data to physical simulation.

### Phase 1: Genomic Discovery (AlphaGenome)
- **Objective**: Identify super-enhancers and oncogenic mutations in patient genomic data.
- **Implementation**: Uses the `AlphaGenomeClient` to query DeepMind's AlphaGenome API (or local heuristics). It compares "normal" vs "mutated" sequences to predict expression scores and epigenetic profiles (H3K27ac, H3K4me1).

### Phase 2: Ligand Engineering (AlphaFold)
- **Objective**: Design high-affinity peptide ligands for target receptors (e.g., EGFR).
- **Implementation**: Automates the generation of JSON job specs for the AlphaFold 3 Server. It parses results to identify the "best" binders based on pLDDT and PAE scores.

### Phase 3: Structure Analysis & Scoring
- **Objective**: Rank and classify binders from Phase 2.
- **Criteria**: Results are classified as `STRONG_BINDER` (pLDDT > 80), `MODERATE`, or `WEAK`.

### Phase 4: Evolutionary Promoter Design (Genetic Algorithm)
- **Objective**: Evolve a synthetic DNA promoter that is highly active in tumor cells but silent in healthy tissue.
- **Innovation**: Implements an **Adaptive Mutation Rate**. If fitness stagnates for 5 generations, the mutation rate increases to escape local optima.
- **Fitness Function**: Weighted combination of Tumor Score, Normal Penalty, and Heat-Shock response elements.

### Phase 5: Thermo-Switch Protein Design
- **Objective**: Engineer a protein switch that unfolds precisely at 43°C (Melting Temp $T_m \approx 40^\circ C$).
- **Physics Model**: Uses a Two-State Native-to-Unfolded thermodynamics model ($ \Delta G = \Delta H - T\Delta S $). It evolves the hydrophobic core of a GCN4 Leucine Zipper scaffold.

### Phase 6: Nanoparticle Surface Topology (Monte Carlo)
- **Objective**: Optimize the distribution of ligands and PEG (stealth) polymers on the nanoparticle surface.
- **Method**: Lattice Monte Carlo using the Metropolis Algorithm. It minimizes steric hindrance and ligand clumping through simulated annealing.

### Phase 7: Biological Circuit Integration
- **Objective**: Model the "AND" gate logic of the therapy.
- **Logic**: 
  - *Input A*: Promoter activity (Cancer context).
  - *Input B*: Switch state (Temperature).
  - *Output*: Kill Signal ($Signal = A \times B$).
- **Verification**: Generates a heatmap showing activation only in the "Tumor + 43°C" quadrant.

### Phase 8: RL-Driven Sequence Design (PPO)
- **Objective**: Train a Reinforcement Learning agent to "write" DNA sequences from scratch.
- **Implementation**: Custom `PromoterDesignEnv` (Gymnasium) where an agent uses Proximal Policy Optimization (PPO) to place nucleotides (A, C, G, T) to maximize biological fitness.

### Phase 9: Physics Verification (OpenMM)
- **Objective**: Validate the thermo-switch unfolding via Molecular Dynamics (MD).
- **Implementation**: Solvates the designed protein in explicit water (TIP3P) and runs simulations at 37°C and 43°C.
- **Acceleration**: Automatically detects and uses **CUDA** or **OpenCL** for GPU-accelerated physics.

### Phase 10: Convergence Visualization
- **Objective**: Generate publication-quality reports.
- **Output**: `evolution_trajectory.png` showing the correlation between fitness breakthroughs and adaptive mutation events.

---

## 🚀 Implementation Guide

### 1. Prerequisites
- **OS**: Windows, Linux, or macOS.
- **Hardware**: NVIDIA GPU (RTX 3070+ recommended) for MD simulations.
- **Environment**: Python 3.10+ and [Miniconda/Anaconda](https://docs.conda.io/en/latest/).

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/bharathvbcr/GenoThermal_Targeting.git
cd Geno-Thermal_Targeting

# Install core dependencies
pip install -r requirements.txt
pip install seaborn pdbfixer gymnasium stable-baselines3

# Install OpenMM with GPU support (via Conda)
conda install -c conda-forge openmm
```

### 3. Running the Master Pipeline
The entire 10-phase workflow is automated via a master orchestrator:
```bash
python run_pipeline.py
```
This script handles dependency checking, handles the hand-off between phases, and generates a centralized `pipeline_master.log`.

### 4. Key Implementation Details
- **Caching**: The `AlphaGenomeClient` implements an in-memory session cache. This ensures that redundant sequences generated during GA or RL training are only scored once, saving significant compute time.
- **PDB Repair**: Since AlphaFold structures often have missing atoms, Phase 9 automatically triggers `pdbfixer` to ensure valid topologies before the MD simulation starts.
- **Logging**: Every script generates its own log file (e.g., `evolver.log`, `physics_verify.log`) for deep debugging.

### 5. Interpreting Results
- **`target_report.json`**: Contains the predicted genomic markers.
- **`evolution_trajectory.png`**: Look for the purple dashed line; it indicates the mutation rate adapting to breakthroughs in fitness.
- **`circuit_heatmap.png`**: Verifies the safety of the therapy; the "NORMAL" row should remain green (low kill signal) even at high temperatures.

---

## 🛠 Advanced Usage: Individual Controls
You can run components independently for research:
- **Optimize NP Surface**: `python hard_mode/nano_topology.py`
- **Train RL Agent**: `python hard_mode/ppo_agent.py`
- **Custom MD Run**: `python hard_mode/physics_verify.py --input simulated_pdbs/your_protein.pdb`
