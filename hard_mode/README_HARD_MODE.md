# Evolutionary Design: Hyperthermia-Gated Synthetic Promoter

This sub-project implements a generative inverse design approach to create a synthetic DNA promoter. Instead of scanning for existing targets, we evolve a sequence that is specifically active only under two conditions:
1.  **Tumor Context:** Presence of cancer-specific transcription factors.
2.  **Heat Shock:** Presence of heat shock factors (HSF1) triggered by magnetic hyperthermia.

## Components

### 1. `evolver.py` (Genetic Algorithm)
*   **Purpose:** Evolves a DNA sequence (promoter) to maximize tumor expression and minimize normal expression.
*   **Mechanism:**
    *   **Population:** Starts with 100 random DNA sequences.
    *   **Fitness Function:** Queries a mock `AlphaGenomeOracle` (simulating the AlphaGenome API) to predict expression in "Tumor" and "Normal" contexts.
    *   **Objective:** `Fitness = (TumorExpression * 1.0) - (NormalExpression * 2.0)`
    *   **Operators:** Uses elitism, tournament selection, single-point crossover, and random mutation.
*   **Output:** A 50bp synthetic DNA sequence optimized for the target criteria.

### 2. `thermo_fold.py` (Coming Soon)
*   **Purpose:** Designs a thermo-labile protein switch (e.g., a modified leucine zipper) that unfolds at 43°C.
*   **Mechanism:** Will use AlphaFold simulations to map pLDDT scores to predicted thermodynamic stability.

## How to Run
1.  Ensure you have the required dependencies:
    ```bash
    pip install numpy
    ```
2.  Run the evolutionary engine:
    ```bash
    python evolver.py
    ```
