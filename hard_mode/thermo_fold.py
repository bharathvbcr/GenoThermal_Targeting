# evolutionary_design/thermo_fold.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
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
        logger.info("ProteinPhysicsOracle init: two-state model, R=%.6f kcal/(mol·K)", R)
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
        logger.debug("_calculate_thermodynamics: base_dH=%.3f, seq_len=%d", delta_H, len(sequence))
        for i in self.core_indices:
            aa = sequence[i]
            delta_H += self.enthalpy_map.get(aa, 0.5) * 2.0
        for i in self.interface_indices:
            aa = sequence[i]
            delta_H += self.enthalpy_map.get(aa, 0.5) * 1.0
        delta_S = self.base_delta_S
        logger.debug("_calculate_thermodynamics: dH=%.3f, dS=%.5f", delta_H, delta_S)
        return delta_H, delta_S

    def predict_melting_temp(self, sequence):
        dH, dS = self._calculate_thermodynamics(sequence)
        if dS == 0:
            logger.debug("predict_melting_temp: dS=0, returning Tm=0")
            return 0
        tm_kelvin = dH / dS
        tm_c = tm_kelvin - 273.15
        logger.debug("predict_melting_temp: dH=%.3f, dS=%.5f, Tm=%.2f°C", dH, dS, tm_c)
        return tm_c

    def predict_folded_fraction(self, sequence, temperature_c):
        temp_k = temperature_c + 273.15
        dH, dS = self._calculate_thermodynamics(sequence)
        delta_G = dH - (temp_k * dS)
        try:
            k_fold = np.exp(-delta_G / (R * temp_k))
            fraction = k_fold / (1.0 + k_fold)
        except OverflowError:
            fraction = 1.0 if delta_G < 0 else 0.0
            logger.debug("predict_folded_fraction: OverflowError at T=%.2f°C, delta_G=%.4f -> fraction=%.1f",
                         temperature_c, delta_G, fraction)
        return fraction

    def predict_plddt(self, sequence, temperature_c):
        fraction = self.predict_folded_fraction(sequence, temperature_c)
        noise = np.random.normal(0, 1.5)
        plddt = min(100.0, max(0.0, 20.0 + (78.0 * fraction) + noise))
        logger.debug("predict_plddt: T=%.1f°C, fraction=%.3f, plddt=%.1f", temperature_c, fraction, plddt)
        return plddt

class ThermoSwitchOptimizer:
    def __init__(self, scaffold):
        logger.info("ThermoSwitchOptimizer init: scaffold='%s...', pop=%d, gens=%d",
                    scaffold[:10], 50, 30)
        self.scaffold = scaffold
        self.oracle = ProteinPhysicsOracle()
        self.population_size = 50
        self.generations = 30
        self.mutation_rate = 0.1

    def mutate(self, sequence):
        seq_list = list(sequence)
        target_indices = [4, 11, 18, 25, 7, 14, 21]
        idx = random.choice(target_indices)
        old_aa = seq_list[idx]
        choices = ['L', 'V', 'I', 'A', 'M', 'F']
        seq_list[idx] = random.choice(choices)
        logger.debug("mutate: pos=%d, %s -> %s", idx, old_aa, seq_list[idx])
        return "".join(seq_list)

    def fitness(self, sequence):
        tm = self.oracle.predict_melting_temp(sequence)
        tm_penalty = abs(tm - TARGET_TM) * 2.0
        plddt_37 = self.oracle.predict_plddt(sequence, TARGET_TEMP_LOW)
        plddt_43 = self.oracle.predict_plddt(sequence, TARGET_TEMP_HIGH)
        switch_score = (plddt_37 - plddt_43)
        if plddt_37 < 75.0:
            logger.debug("fitness: pLDDT@37C=%.1f < 75 threshold, applying -50 penalty", plddt_37)
            switch_score -= 50.0
        final_score = switch_score - tm_penalty
        logger.debug("fitness: Tm=%.2f, plddt37=%.1f, plddt43=%.1f, switch=%.2f, final=%.2f",
                     tm, plddt_37, plddt_43, switch_score, final_score)
        return final_score, tm

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
            improved = current_best[0] > best_score
            if improved:
                best_score = current_best[0]
                best_overall = current_best
            logger.debug("Gen %02d: best_score=%.2f, Tm=%.1f\u00b0C, improved=%s",
                         gen, current_best[0], current_best[2], improved)
            if gen % 5 == 0:
                logger.info("Gen %02d: Best Score=%.2f | Tm=%.1f\u00b0C",
                            gen, current_best[0], current_best[2])
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
        os.makedirs("outputs/figures", exist_ok=True)
        plt.savefig("outputs/figures/thermo_profile.png")
        logger.info(f"Melting curve saved to 'outputs/figures/thermo_profile.png'")

if __name__ == "__main__":
    designer = ThermoSwitchOptimizer(BASE_SCAFFOLD)
    best_seq, best_tm = designer.run()
    logger.info("--- Design Complete ---")
    logger.info(f"Optimal Sequence: {best_seq}")
    logger.info(f"Predicted Tm:     {best_tm:.2f}\u00b0C")
    designer.plot_melting_curve(best_seq, best_tm)