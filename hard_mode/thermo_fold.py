# evolutionary_design/thermo_fold.py

import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("thermo_fold.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ThermoFold")

# --- CONFIGURATION ---
TARGET_TEMP_LOW = 37.0   # Body temperature (Stable)
TARGET_TEMP_HIGH = 43.0  # Hyperthermia (Unfolding)
TARGET_TM = 40.0         # Ideal Melting Temp (The "Switch" Point)

# Boltzmann Constant (kcal/mol*K)
R = 0.001987 

# GCN4 Leucine Zipper Scaffold (Heptad repeat: abcdefg)
BASE_SCAFFOLD = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"

class ProteinPhysicsOracle:
    """
    Simulates protein thermodynamics.
    Uses a simplified Two-State Model: Native <-> Unfolded.
    """
    def __init__(self):
        self.enthalpy_map = {
            'L': -3.5, 'I': -3.4, 'V': -2.8, 'M': -2.9,
            'A': -1.5, 'F': -3.0, 'Y': -2.5, 'W': -2.8,
            'K': 0.0, 'E': 0.0, 'R': 0.0, 'D': 0.0, 'Q': 0.0, 'N': 0.0,
            'G': 1.0, 'P': 2.0
        }
        self.base_delta_H = -20.0
        self.base_delta_S = -0.15
        self.core_indices = [4, 11, 18, 25]
        self.interface_indices = [0, 7, 14, 21, 28]

    def _calculate_thermodynamics(self, sequence):
        delta_H = self.base_delta_H
        for i in self.core_indices:
            aa = sequence[i]
            delta_H += self.enthalpy_map.get(aa, 0.5) * 2.0
        for i in self.interface_indices:
            aa = sequence[i]
            delta_H += self.enthalpy_map.get(aa, 0.5) * 1.0
        delta_S = self.base_delta_S
        return delta_H, delta_S

    def predict_melting_temp(self, sequence):
        dH, dS = self._calculate_thermodynamics(sequence)
        if dS == 0: return 0
        tm_kelvin = dH / dS
        return tm_kelvin - 273.15

    def predict_folded_fraction(self, sequence, temperature_c):
        temp_k = temperature_c + 273.15
        dH, dS = self._calculate_thermodynamics(sequence)
        delta_G = dH - (temp_k * dS)
        try:
            k_fold = np.exp(-delta_G / (R * temp_k))
            fraction = k_fold / (1.0 + k_fold)
        except OverflowError:
            fraction = 1.0 if delta_G < 0 else 0.0
        return fraction

    def predict_plddt(self, sequence, temperature_c):
        fraction = self.predict_folded_fraction(sequence, temperature_c)
        noise = np.random.normal(0, 1.5)
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
        seq_list = list(sequence)
        target_indices = [4, 11, 18, 25, 7, 14, 21] 
        idx = random.choice(target_indices)
        choices = ['L', 'V', 'I', 'A', 'M', 'F']
        seq_list[idx] = random.choice(choices)
        return "".join(seq_list)

    def fitness(self, sequence):
        tm = self.oracle.predict_melting_temp(sequence)
        tm_penalty = abs(tm - TARGET_TM) * 2.0
        plddt_37 = self.oracle.predict_plddt(sequence, TARGET_TEMP_LOW)
        plddt_43 = self.oracle.predict_plddt(sequence, TARGET_TEMP_HIGH)
        switch_score = (plddt_37 - plddt_43)
        if plddt_37 < 75.0: switch_score -= 50.0
        return switch_score - tm_penalty, tm

    def run(self):
        logger.info(f"--- Starting Protein Thermo-Switch Design ---")
        population = [self.scaffold] + [self.mutate(self.scaffold) for _ in range(self.population_size - 1)]
        best_overall = None
        best_score = -float('inf')
        for gen in range(self.generations):
            scored_pop = []
            for seq in population:
                score, tm = self.fitness(seq)
                scored_pop.append((score, seq, tm))
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            current_best = scored_pop[0]
            if current_best[0] > best_score:
                best_score = current_best[0]
                best_overall = current_best
            if gen % 5 == 0:
                logger.info(f"Gen {gen}: Best Score={current_best[0]:.2f} | Tm={current_best[2]:.1f}\u00b0C")
            survivors = [x[1] for x in scored_pop[:int(self.population_size * 0.2)]]
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
        plt.plot(temps, fractions, label='Theoretical Folded %', color='blue', alpha=0.6, linestyle='--')
        plt.scatter(temps, plddts, s=10, color='black', alpha=0.5, label='Simulated AlphaFold pLDDT')
        plt.plot(temps, plddts, color='orange', linewidth=2, alpha=0.8)
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
        logger.info(f"Melting curve saved to 'thermo_profile.png'")

if __name__ == "__main__":
    designer = ThermoSwitchOptimizer(BASE_SCAFFOLD)
    best_seq, best_tm = designer.run()
    logger.info("--- Design Complete ---")
    logger.info(f"Optimal Sequence: {best_seq}")
    logger.info(f"Predicted Tm:     {best_tm:.2f}\u00b0C")
    designer.plot_melting_curve(best_seq, best_tm)