import os
import numpy as np
import matplotlib.pyplot as plt
import random
import logging

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nano_topology.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NanoTopology")

GRID_SIZE = 50
STEPS = 50000
TEMP_SIM = 1.0
DENSITY_LIGAND = 0.2
DENSITY_PEG = 0.3

EMPTY = 0
LIGAND = 1
PEG = 2

class NanoTopologySim:
    def __init__(self):
        logger.info("NanoTopologySim init: grid=%dx%d, steps=%d, T=%.2f, "
                    "ligand_density=%.2f, peg_density=%.2f",
                    GRID_SIZE, GRID_SIZE, STEPS, TEMP_SIM, DENSITY_LIGAND, DENSITY_PEG)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.temp_sim = TEMP_SIM
        self._initialize_grid()

    def _initialize_grid(self):
        n_ligands = int(GRID_SIZE * GRID_SIZE * DENSITY_LIGAND)
        n_pegs = int(GRID_SIZE * GRID_SIZE * DENSITY_PEG)
        logger.info("_initialize_grid: placing %d ligands and %d PEG molecules on %d sites.",
                    n_ligands, n_pegs, GRID_SIZE * GRID_SIZE)
        coords = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(coords)
        for i in range(n_ligands):
            r, c = coords[i]
            self.grid[r, c] = LIGAND
        for i in range(n_ligands, n_ligands + n_pegs):
            r, c = coords[i]
            self.grid[r, c] = PEG
        logger.info("Grid initialized: %d ligand, %d PEG, %d empty sites.",
                    n_ligands, n_pegs, GRID_SIZE * GRID_SIZE - n_ligands - n_pegs)

    def calculate_energy(self, r, c):
        val = self.grid[r, c]
        if val == EMPTY:
            return 0
        energy = 0
        neighbors = [((r+1)%GRID_SIZE, c), ((r-1)%GRID_SIZE, c), (r, (c+1)%GRID_SIZE), (r, (c-1)%GRID_SIZE)]
        for nr, nc in neighbors:
            n_val = self.grid[nr, nc]
            if n_val == EMPTY:
                continue
            if val == LIGAND and n_val == LIGAND:
                energy += 1.0
            elif val == PEG and n_val == PEG:
                energy += 0.5
            elif val != n_val:
                energy -= 0.5
        return energy

    def run_annealing(self):
        logger.info("--- Starting Nano-Topology Simulation (Lattice Monte Carlo) ---")
        logger.info("Grid=%dx%d, Steps=%d, initial T=%.3f", GRID_SIZE, GRID_SIZE, STEPS, self.temp_sim)
        n_accepted = 0
        n_rejected = 0
        for i in range(STEPS):
            r1, c1 = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            dr, dc = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
            r2, c2 = (r1+dr)%GRID_SIZE, (c1+dc)%GRID_SIZE
            e_before = self.calculate_energy(r1, c1) + self.calculate_energy(r2, c2)
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            e_after = self.calculate_energy(r1, c1) + self.calculate_energy(r2, c2)
            delta_E = e_after - e_before
            if delta_E > 0:
                if random.random() > np.exp(-delta_E / self.temp_sim):
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    n_rejected += 1
                else:
                    n_accepted += 1
            else:
                n_accepted += 1
            if i % 1000 == 0:
                self.temp_sim *= 0.99
            if i > 0 and i % 10000 == 0:
                logger.info("Step %d/%d (%.0f%%) | T=%.4f | accepted=%d, rejected=%d",
                            i, STEPS, 100.0 * i / STEPS, self.temp_sim, n_accepted, n_rejected)
        logger.info("Simulation complete. Final T=%.4f | total accepted=%d, rejected=%d",
                    self.temp_sim, n_accepted, n_rejected)

    def plot_surface(self):
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.colors.ListedColormap(['white', 'crimson', 'dodgerblue'])
        plt.imshow(self.grid, cmap=cmap, interpolation='nearest')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='crimson', label='Ligand (GE11)'),
            Patch(facecolor='dodgerblue', label='Stealth (PEG)'),
            Patch(facecolor='white', edgecolor='gray', label='Empty Surface')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(f"Optimized Nanoparticle Surface Topology")
        plt.axis('off')
        os.makedirs("outputs/figures", exist_ok=True)
        plt.savefig("outputs/figures/nano_surface.png")
        logger.info("Surface topology map saved to 'outputs/figures/nano_surface.png'")

if __name__ == "__main__":
    sim = NanoTopologySim()
    sim.run_annealing()
    sim.plot_surface()