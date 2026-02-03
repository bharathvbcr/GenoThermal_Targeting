# evolutionary_design/thermo_fold.py

import random
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- CONFIGURATION ---
TARGET_TEMP_LOW = 37.0   # Body temperature (Stable)
TARGET_TEMP_HIGH = 43.0  # Hyperthermia (Unfolding)
TARGET_TM = 40.0         # Ideal Melting Temp (The "Switch" Point)

# Boltzmann Constant (kcal/mol*K)
R = 0.001987 

# GCN4 Leucine Zipper Scaffold (Heptad repeat: abcdefg)
# We focus on 'd' (Core, hydrophobic) and 'a' (Core/Interface) positions.
# Scaffold: MKQLEDK VEELLSK NYHLENE VARLKKL VGER
# Indices:  0123456 7890123 4567890 1234567 8901
# 'd' positions: 4, 11, 18, 25 (Primary Core)
# 'a' positions: 0, 7, 14, 21, 28 (Secondary Core/Salt bridges)
BASE_SCAFFOLD = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"

class ProteinPhysicsOracle:
    """
    Simulates protein thermodynamics.
    Uses a simplified Two-State Model: Native <-> Unfolded.
    DeltaG = DeltaH - T * DeltaS
    """
    def __init__(self):
        # Contribution to Enthalpy (DeltaH) - Interaction Strength (Negative is stable)
        # Simplified scale based on packing efficiency in a coiled-coil core
        self.enthalpy_map = {
            'L': -3.5, 'I': -3.4, 'V': -2.8, 'M': -2.9, # Strong core formers
            'A': -1.5, # Weak core
            'F': -3.0, 'Y': -2.5, 'W': -2.8, # Bulky
            'K': 0.0, 'E': 0.0, 'R': 0.0, 'D': 0.0, 'Q': 0.0, 'N': 0.0, # Charged/Polar (bad for core)
            'G': 1.0, 'P': 2.0 # Destabilizing
        }
        
        # Base Enthalpy/Entropy for the scaffold (excluding core contributions)
        self.base_delta_H = -20.0 # kcal/mol (Hydrogen bonds, backbone)
        self.base_delta_S = -0.15 # kcal/mol*K (Entropy cost of folding)
        
        self.core_indices = [4, 11, 18, 25] # d-positions
        self.interface_indices = [0, 7, 14, 21, 28] # a-positions

    def _calculate_thermodynamics(self, sequence):
        """Estimate DeltaH and DeltaS for the sequence."""
        delta_H = self.base_delta_H
        
        # Sum contributions from Core residues ('d' positions)
        for i in self.core_indices:
            aa = sequence[i]
            delta_H += self.enthalpy_map.get(aa, 0.5) * 2.0 # Weight for primary core
            
        # Sum contributions from Interface residues ('a' positions)
        for i in self.interface_indices:
            aa = sequence[i]
            # 'a' positions are less strictly hydrophobic, but packing matters
            delta_H += self.enthalpy_map.get(aa, 0.5) * 1.0

        # Estimate Entropy (DeltaS)
        # Assuming mutations don't drastically change unfolded state entropy 
        # unless Glycine/Proline are introduced.
        delta_S = self.base_delta_S
        
        return delta_H, delta_S

    def predict_melting_temp(self, sequence):
        """Tm = DeltaH / DeltaS (in Kelvin)"""
        dH, dS = self._calculate_thermodynamics(sequence)
        if dS == 0: return 0
        tm_kelvin = dH / dS
        return tm_kelvin - 273.15 # Convert to Celsius

    def predict_folded_fraction(self, sequence, temperature_c):
        """
        P_folded = 1 / (1 + K_eq) = 1 / (1 + exp(-DeltaG / RT))
        """
        temp_k = temperature_c + 273.15
        dH, dS = self._calculate_thermodynamics(sequence)
        delta_G = dH - (temp_k * dS)
        
        try:
            # K_eq = [Native] / [Unfolded] ... wait, standard is U/N for exp(-dG/RT) usually defined U-N
            # Let's use DeltaG_folding = G_native - G_unfolded (should be negative for stability)
            # K_folding = exp(-DeltaG / RT)
            # Fraction Folded = K / (1+K)
            k_fold = np.exp(-delta_G / (R * temp_k))
            fraction = k_fold / (1.0 + k_fold)
        except OverflowError:
            fraction = 1.0 if delta_G < 0 else 0.0
            
        return fraction

    def predict_plddt(self, sequence, temperature_c):
        """Maps folded fraction to AlphaFold pLDDT score."""
        fraction = self.predict_folded_fraction(sequence, temperature_c)
        # Add noise
        noise = np.random.normal(0, 1.5)
        # Map 0.0-1.0 to 20-98
        plddt = 20.0 + (78.0 * fraction)
        return min(100.0, max(0.0, plddt + noise))

class ThermoSwitchOptimizer:
    def __init__(self, scaffold):
        self.scaffold = scaffold
        self.oracle = ProteinPhysicsOracle()
        self.population_size = 50
        self.generations = 30
        self.mutation_rate = 0.1

    def mutate(self, sequence):
        """Mutate hydrophobic core to tune stability."""
        seq_list = list(sequence)
        # Target d-positions and a-positions
        target_indices = [4, 11, 18, 25, 7, 14, 21] 
        
        idx = random.choice(target_indices)
        # Allow substitutions that fine-tune packing (L, V, I, A, M)
        current_aa = seq_list[idx]
        choices = ['L', 'V', 'I', 'A', 'M', 'F']
        # Bias: If current is L, try V or A to destabilize
        
        seq_list[idx] = random.choice(choices)
        return "".join(seq_list)

    def fitness(self, sequence):
        """
        Objective: 
        1. Tm should be close to TARGET_TM (40C).
        2. Steep transition (cooperativity) - inherent in the physics model.
        3. High stability at 37C (pLDDT > 80), Low at 43C (pLDDT < 60).
        """
        tm = self.oracle.predict_melting_temp(sequence)
        
        # Penalize distance from target Tm
        tm_penalty = abs(tm - TARGET_TM) * 2.0
        
        # Check actual switching behavior
        plddt_37 = self.oracle.predict_plddt(sequence, TARGET_TEMP_LOW)
        plddt_43 = self.oracle.predict_plddt(sequence, TARGET_TEMP_HIGH)
        
        switch_score = (plddt_37 - plddt_43) # Maximizing the gap
        
        # Constraints
        if plddt_37 < 75.0: switch_score -= 50.0 # Must be folded at body temp
        
        return switch_score - tm_penalty, tm

    def run(self):
        print(f"--- Starting Protein Thermo-Switch Design ---")
        print(f"Algorithm: Genetic Search | Target Tm: {TARGET_TM}\u00b0C")
        
        # Init population
        population = [self.scaffold] + [self.mutate(self.scaffold) for _ in range(self.population_size - 1)]
        
        best_overall = None
        best_score = -float('inf')
        
        for gen in range(self.generations):
            scored_pop = []
            for seq in population:
                score, tm = self.fitness(seq)
                scored_pop.append((score, seq, tm))
            
            # Sort
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            current_best = scored_pop[0]
            
            if current_best[0] > best_score:
                best_score = current_best[0]
                best_overall = current_best
            
            # Logging
            if gen % 5 == 0:
                print(f"Gen {gen}: Best Score={current_best[0]:.2f} | Tm={current_best[2]:.1f}\u00b0C")
                print(f"       Seq: {current_best[1]}")
            
            # Selection (Top 20%)
            survivors = [x[1] for x in scored_pop[:int(self.population_size * 0.2)]]
            
            # Repopulate
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                parent = random.choice(survivors)
                new_pop.append(self.mutate(parent))
            
            population = new_pop
            
        return best_overall[1], best_overall[2]

    def plot_melting_curve(self, sequence, tm_val):
        temps = np.arange(25, 55, 0.5)
        fractions = [self.oracle.predict_folded_fraction(sequence, t) * 100 for t in temps]
        plddts = [self.oracle.predict_plddt(sequence, t) for t in temps]
        
        plt.figure(figsize=(9, 6))
        
        # Plot Folded Fraction (Theoretical)
        plt.plot(temps, fractions, label='Theoretical Folded %', color='blue', alpha=0.6, linestyle='--')
        
        # Plot pLDDT (Simulated AlphaFold)
        plt.scatter(temps, plddts, s=10, color='black', alpha=0.5, label='Simulated AlphaFold pLDDT')
        
        # Smooth line for pLDDT
        plt.plot(temps, plddts, color='orange', linewidth=2, alpha=0.8)

        # Annotations
        plt.axvline(x=TARGET_TEMP_LOW, color='green', linestyle='-', alpha=0.5, label='Body (37\u00b0C)')
        plt.axvline(x=TARGET_TEMP_HIGH, color='red', linestyle='-', alpha=0.5, label='Hyperthermia (43\u00b0C)')
        plt.axvline(x=tm_val, color='purple', linestyle=':', label=f'Designed Tm ({tm_val:.1f}\u00b0C)')
        
        plt.fill_between(temps, 0, 100, where=(temps >= 37) & (temps <= 43), color='yellow', alpha=0.1, label='Switch Window')

        plt.title(f"Designed Thermo-Switch: {sequence[:10]}...")
        plt.xlabel("Temperature (\u00b0C)")
        plt.ylabel("Protein Stability (pLDDT / % Folded)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("thermo_profile.png")
        print(f"\nMelting curve saved to 'thermo_profile.png'")

if __name__ == "__main__":
    designer = ThermoSwitchOptimizer(BASE_SCAFFOLD)
    best_seq, best_tm = designer.run()
    
    print("\n--- Design Complete ---")
    print(f"Optimal Sequence: {best_seq}")
    print(f"Predicted Tm:     {best_tm:.2f}\u00b0C")
    
    designer.plot_melting_curve(best_seq, best_tm)

