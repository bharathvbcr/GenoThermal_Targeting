# evolutionary_design/evolver.py
#
# Genetic Algorithm for evolving synthetic DNA promoters.
# The GA loop runs locally (lightweight string ops + arithmetic).
# The fitness oracle dispatches to the AlphaGenome API when available,
# falling back to local motif scanning for dev/CI.

import random
import numpy as np
import time
import threading
import concurrent.futures
import re
import csv
import os
import sys
import logging
import functools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evolver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evolver")

# --- Configuration ---
POPULATION_SIZE = 100
GENOME_LENGTH = 200  # 200bp synthetic promoter
GENERATIONS = 50
INITIAL_MUTATION_RATE = 0.05
ELITISM_COUNT = 10
STAGNATION_THRESHOLD = 5  # Generations without improvement to trigger adaptation

# API rate-limiting: max concurrent workers and delay between calls
API_MAX_WORKERS = 3       # Limit parallel API calls
API_CALL_DELAY = 1.0      # Seconds between API submissions
LOCAL_MAX_WORKERS = None   # Unlimited for local fallback (CPU-bound)
LOG_FILE = "evolution_log.csv"

# Weights for Fitness Function
WEIGHT_TUMOR = 1.5
WEIGHT_NORMAL = 2.0  # High penalty for off-target expression
WEIGHT_HEAT = 1.2    # Bonus for thermal sensitivity
PENALTY_GC_DEVIATION = 0.5

# Attempt to import the project's API client
try:
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from alphagenome_utils import AlphaGenomeClient as _AGClient
    _AG_AVAILABLE = True
except ImportError:
    _AG_AVAILABLE = False

class AlphaGenomeOracle:
    """
    Fitness evaluator that wraps the AlphaGenome API.
    """
    def __init__(self, api_key=None, mode="Auto"):
        self._api_client = None
        force_local = (mode == "Local")
        
        if _AG_AVAILABLE:
            self._api_client = _AGClient(api_key=api_key, force_local=force_local)
            if self._api_client._mode == "API":
                logger.info("Oracle: Using AlphaGenome API for fitness evaluation.")
            else:
                logger.info("Oracle: Using local motif scanner.")
                self._api_client = None

        self.tumor_motifs = [r"AGAACA", r"GGATCTT", r"CACGTG"]
        self.normal_motifs = [r"TATAAA", r"CCAAT", r"GCGCGC"]
        self.heat_motifs = [r"GAA..TTC", r"TTC..GAA"]
        
    def _scan_motifs(self, sequence, motif_list):
        count = 0
        for pattern in motif_list:
            count += len(re.findall(pattern, sequence))
        return count

    def evaluate_sequence_properties(self, sequence):
        tumor_hits = self._scan_motifs(sequence, self.tumor_motifs)
        normal_hits = self._scan_motifs(sequence, self.normal_motifs)
        heat_hits = self._scan_motifs(sequence, self.heat_motifs)
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / max(len(sequence), 1)

        return {
            "tumor_score": tumor_hits * 20.0,
            "normal_score": normal_hits * 15.0,
            "heat_score": heat_hits * 25.0,
            "gc_penalty": abs(0.55 - gc_content) * 100.0,
            "raw_counts": (tumor_hits, normal_hits, heat_hits),
        }

    def score(self, sequence):
        if self._api_client is not None:
            return self._api_client.predict_sequence_fitness(sequence)
        return self._local_score(sequence)

    def _local_score(self, sequence):
        props = self.evaluate_sequence_properties(sequence)
        score = (props["tumor_score"] * WEIGHT_TUMOR) - \
                (props["normal_score"] * WEIGHT_NORMAL) + \
                (props["heat_score"] * WEIGHT_HEAT) - \
                (props["gc_penalty"] * PENALTY_GC_DEVIATION)
        score += random.uniform(-1, 1)
        return max(0.0, score)

def calculate_fitness(sequence, oracle):
    score = oracle.score(sequence)
    props = oracle.evaluate_sequence_properties(sequence)
    return max(0.0, score), props

class GeneticOptimizer:
    def __init__(self, oracle):
        self.oracle = oracle
        self.population = self._initialize_population()
        self.history = []
        self._uses_api = (oracle._api_client is not None)
        self.mutation_rate = INITIAL_MUTATION_RATE
        self.stagnation_counter = 0
        self.best_fitness_all_time = -float('inf')

    def _initialize_population(self):
        bases = ['A', 'C', 'G', 'T']
        return [''.join(random.choices(bases, k=GENOME_LENGTH)) for _ in range(POPULATION_SIZE)]

    def mutate(self, sequence):
        seq_list = list(sequence)
        num_mutations = max(1, int(len(sequence) * self.mutation_rate))
        indices = random.sample(range(len(sequence)), num_mutations)
        for i in indices:
            seq_list[i] = random.choice(['A', 'C', 'G', 'T'])
        return "".join(seq_list)

    def crossover(self, parent1, parent2):
        pt1 = random.randint(1, GENOME_LENGTH - 2)
        pt2 = random.randint(pt1 + 1, GENOME_LENGTH - 1)
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        return child1, child2

    def check_convergence(self, current_best_fitness):
        if current_best_fitness > self.best_fitness_all_time:
            self.best_fitness_all_time = current_best_fitness
            self.stagnation_counter = 0
            # Reset mutation rate if we broke through
            if self.mutation_rate > INITIAL_MUTATION_RATE:
                logger.info("Fitness improved! Resetting mutation rate.")
                self.mutation_rate = INITIAL_MUTATION_RATE
        else:
            self.stagnation_counter += 1
            
        if self.stagnation_counter >= STAGNATION_THRESHOLD:
            # Increase mutation rate to escape local optima
            old_rate = self.mutation_rate
            self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
            if self.mutation_rate != old_rate:
                logger.info(f"Stagnation detected ({self.stagnation_counter} gens). Increasing mutation rate: {old_rate:.3f} -> {self.mutation_rate:.3f}")
            self.stagnation_counter = 0 # Reset counter to give new rate time to work

    def run(self):
        logger.info(f"--- Starting Evolutionary Design (Hard Mode) ---")
        logger.info(f"Goal: Hyperthermia-Gated Prostate Cancer Promoter")
        logger.info(f"Params: Gen={GENERATIONS} | Pop={POPULATION_SIZE} | Len={GENOME_LENGTH}bp")
        
        with open(LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Generation", "Best_Fitness", "Tumor_Score", "Normal_Score", "Heat_Score", "Mutation_Rate"])

        for gen in range(GENERATIONS):
            max_w = API_MAX_WORKERS if self._uses_api else LOCAL_MAX_WORKERS
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                future_to_seq = {}
                for i, seq in enumerate(self.population):
                    future_to_seq[executor.submit(calculate_fitness, seq, self.oracle)] = seq
                    if self._uses_api and i % API_MAX_WORKERS == (API_MAX_WORKERS - 1):
                        time.sleep(API_CALL_DELAY)
                for future in concurrent.futures.as_completed(future_to_seq):
                    seq = future_to_seq[future]
                    results[seq] = future.result()

            ranked_pop = sorted(self.population, key=lambda s: results[s][0], reverse=True)
            best_seq = ranked_pop[0]
            best_fit, best_props = results[best_seq]
            
            # Check for stagnation
            self.check_convergence(best_fit)
            
            self.history.append((gen, best_fit))
            with open(LOG_FILE, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([gen, f"{best_fit:.2f}", 
                                 best_props["tumor_score"], 
                                 best_props["normal_score"], 
                                 best_props["heat_score"],
                                 f"{self.mutation_rate:.3f}"])
            
            if gen % 10 == 0 or gen == GENERATIONS - 1:
                logger.info(f"Gen {gen:02}: Best Fitness={best_fit:.2f} | MutRate={self.mutation_rate:.3f}")
                logger.info(f"       [Scores] Tumor: {best_props['tumor_score']:.1f} | Normal: {best_props['normal_score']:.1f} | Heat: {best_props['heat_score']:.1f}")
                logger.debug(f"       [Seq Start] {best_seq[:30]}...")

            new_pop = ranked_pop[:ELITISM_COUNT]
            while len(new_pop) < POPULATION_SIZE:
                tournament = random.sample(ranked_pop[:50], 5)
                parent1 = max(tournament, key=lambda s: results[s][0])
                tournament = random.sample(ranked_pop[:50], 5)
                parent2 = max(tournament, key=lambda s: results[s][0])
                c1, c2 = self.crossover(parent1, parent2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < POPULATION_SIZE:
                    new_pop.append(self.mutate(c2))
            self.population = new_pop

        return ranked_pop[0], self.history

if __name__ == "__main__":
    oracle = AlphaGenomeOracle()
    optimizer = GeneticOptimizer(oracle)
    best_promoter, history = optimizer.run()
    logger.info("--- Optimization Complete ---")
    logger.info(f"Final Synthetic Promoter: {best_promoter}")
    logger.info(f"Evolution log saved to {LOG_FILE}")
