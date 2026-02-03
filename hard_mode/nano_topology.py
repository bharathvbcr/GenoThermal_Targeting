import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION (Nightmare Mode) ---
GRID_SIZE = 50       # 50x50nm patch on nanoparticle surface
STEPS = 50000        # Monte Carlo steps
TEMP_SIM = 1.0       # Simulation 'Temperature' (for annealing, not physical temp)
DENSITY_LIGAND = 0.2 # 20% coverage target
DENSITY_PEG = 0.3    # 30% coverage (Stealth polymer)

# States
EMPTY = 0
LIGAND = 1  # GE11
PEG = 2     # Polyethylene Glycol (Stealth)

class NanoTopologySim:
    """
    Simulates the stochastic self-assembly of ligands on a nanoparticle surface.
    Uses a Lattice Monte Carlo method (Metropolis Algorithm).
    Objective: Minimize steric hindrance while maximizing uniform distribution.
    """
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.temp_sim = TEMP_SIM
        self._initialize_grid()
        
    def _initialize_grid(self):
        # Randomly populate grid
        n_ligands = int(GRID_SIZE * GRID_SIZE * DENSITY_LIGAND)
        n_pegs = int(GRID_SIZE * GRID_SIZE * DENSITY_PEG)
        
        coords = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(coords)
        
        for i in range(n_ligands):
            r, c = coords[i]
            self.grid[r, c] = LIGAND
            
        for i in range(n_ligands, n_ligands + n_pegs):
            r, c = coords[i]
            self.grid[r, c] = PEG

    def calculate_energy(self, r, c):
        """
        Hamiltonian Function:
        E = +1.0 for Ligand-Ligand neighbors (Clumping - BAD)
        E = +0.5 for PEG-PEG neighbors (Clumping - BAD)
        E = -0.5 for Ligand-PEG neighbors (Mixing - GOOD)
        """
        val = self.grid[r, c]
        if val == EMPTY:
            return 0
        
        energy = 0
        neighbors = [
            ((r+1)%GRID_SIZE, c), ((r-1)%GRID_SIZE, c),
            (r, (c+1)%GRID_SIZE), (r, (c-1)%GRID_SIZE)
        ]
        
        for nr, nc in neighbors:
            n_val = self.grid[nr, nc]
            if n_val == EMPTY:
                continue
            
            if val == LIGAND and n_val == LIGAND:
                energy += 1.0 # Clumping penalty
            elif val == PEG and n_val == PEG:
                energy += 0.5 # PEG clumping is less bad
            elif val != n_val:
                energy -= 0.5 # Favorable mixing
                
        return energy

    def run_annealing(self):
        print(f"--- Starting Nano-Topology Simulation (Lattice Monte Carlo) ---")
        print(f"Grid: {GRID_SIZE}x{GRID_SIZE} | Ligand Density: {DENSITY_LIGAND}")
        
        current_energy = 0
        # Calculate initial system energy
        # (Simplified: logic usually done per move)
        
        for i in range(STEPS):
            # Pick a random site
            r1, c1 = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            # Pick a neighbor to swap with
            dr, dc = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
            r2, c2 = (r1+dr)%GRID_SIZE, (c1+dc)%GRID_SIZE
            
            # Calculate Local Energy BEFORE swap
            e_before = self.calculate_energy(r1, c1) + self.calculate_energy(r2, c2)
            
            # Tentative Swap
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            
            # Calculate Local Energy AFTER swap
            e_after = self.calculate_energy(r1, c1) + self.calculate_energy(r2, c2)
            
            delta_E = e_after - e_before
            
            # Metropolis Criterion
            if delta_E > 0:
                # If energy increases (bad), accept with probability P = exp(-dE/T)
                if random.random() > np.exp(-delta_E / self.temp_sim):
                    # Reject: Swap back
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            
            # Annealing Schedule: Cool down
            if i % 1000 == 0:
                self.temp_sim *= 0.99
                
        print("Simulation Complete. Final State Reached.")

    def plot_surface(self):
        plt.figure(figsize=(8, 8))
        # Custom colormap: White (Empty), Red (Ligand), Blue (PEG)
        cmap = plt.cm.colors.ListedColormap(['white', 'crimson', 'dodgerblue'])
        plt.imshow(self.grid, cmap=cmap, interpolation='nearest')
        
        # Create legend elements manually
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='crimson', label='Ligand (GE11)'),
            Patch(facecolor='dodgerblue', label='Stealth (PEG)'),
            Patch(facecolor='white', edgecolor='gray', label='Empty Surface')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Optimized Nanoparticle Surface Topology\n(Minimizing Steric Hindrance)")
        plt.axis('off')
        plt.savefig("nano_surface.png")
        print("Surface topology map saved to 'nano_surface.png'")

if __name__ == "__main__":
    sim = NanoTopologySim()
    sim.run_annealing()
    sim.plot_surface()
