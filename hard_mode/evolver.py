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
_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evolver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evolver")

# --- Configuration ---
# Smoke mode (GENOTHERMAL_SMOKE=1) shrinks the GA for a fast end-to-end demo.
_SMOKE = bool(os.environ.get("GENOTHERMAL_SMOKE"))
if _SMOKE:
    logger.info("SMOKE MODE active (GENOTHERMAL_SMOKE=1): pop=%d, gens=%d", 8, 3)
POPULATION_SIZE = 8 if _SMOKE else 100
GENOME_LENGTH = 200  # 200bp synthetic promoter
GENERATIONS = 3 if _SMOKE else 50
INITIAL_MUTATION_RATE = 0.05
ELITISM_COUNT = 2 if _SMOKE else 10
STAGNATION_THRESHOLD = 5  # Generations without improvement to trigger adaptation

# API rate-limiting: max concurrent workers and delay between calls
API_MAX_WORKERS = 3       # Limit parallel API calls
API_CALL_DELAY = 1.0      # Seconds between API submissions
LOCAL_MAX_WORKERS = None   # Unlimited for local fallback (CPU-bound)
LOG_FILE = "outputs/reports/evolution_log.csv"

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
    logger.warning("alphagenome_utils not found — oracle will use local motif scanner only.")
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
        props = {
            "tumor_score": tumor_hits * 20.0,
            "normal_score": normal_hits * 15.0,
            "heat_score": heat_hits * 25.0,
            "gc_penalty": abs(0.55 - gc_content) * 100.0,
            "raw_counts": (tumor_hits, normal_hits, heat_hits),
        }
        logger.debug("evaluate_sequence_properties: tumor=%d, normal=%d, heat=%d, gc=%.3f",
                     tumor_hits, normal_hits, heat_hits, gc_content)
        return props

    def score(self, sequence):
        logger.debug("AlphaGenomeOracle.score: api=%s, seq=%s...", self._api_client is not None, sequence[:8])
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
        result = max(0.0, score)
        logger.debug("_local_score: %.4f (tumor=%.1f, normal=%.1f, heat=%.1f)",
                     result, props["tumor_score"], props["normal_score"], props["heat_score"])
        return result

def calculate_fitness(sequence, oracle):
    logger.debug("calculate_fitness: seq=%s...", sequence[:10])
    score = oracle.score(sequence)
    props = oracle.evaluate_sequence_properties(sequence)
    fitness = max(0.0, score)
    logger.debug("calculate_fitness result: %.4f", fitness)
    return fitness, props

class GeneticOptimizer:
    def __init__(self, oracle):
        logger.info("GeneticOptimizer.__init__: oracle=%s, pop=%d, gens=%d, genome_len=%d",
                    type(oracle).__name__, POPULATION_SIZE, GENERATIONS, GENOME_LENGTH)
        self.oracle = oracle
        self.population = self._initialize_population()
        self.history = []
        self._uses_api = (oracle._api_client is not None)
        self.mutation_rate = INITIAL_MUTATION_RATE
        self.stagnation_counter = 0
        self.best_fitness_all_time = -float('inf')
        # Flash fan-out: opt-in via GENOTHERMAL_FLASH=1 and an importable endpoint.
        self._flash_enabled = self._init_flash()
        self._flash_metrics = None  # FanoutMetrics, lazily created on first Flash generation
        logger.info("GeneticOptimizer ready: uses_api=%s, flash=%s", self._uses_api, self._flash_enabled)

    def _init_flash(self):
        if not os.environ.get("GENOTHERMAL_FLASH"):
            logger.debug("_init_flash: GENOTHERMAL_FLASH not set — Flash disabled.")
            return False
        logger.info("_init_flash: GENOTHERMAL_FLASH=1 detected, attempting Flash import.")
        try:
            import flash_fitness
            if flash_fitness.FLASH_AVAILABLE:
                logger.info("Flash fan-out ENABLED: scoring on the RunPod Flash fleet (workers 0->50).")
                return True
        except ImportError:
            logger.warning("_init_flash: flash_fitness import failed — falling back to local ThreadPoolExecutor.")
        logger.info("Flash requested but unavailable; using local ThreadPoolExecutor.")
        return False

    def _evaluate_population(self, population):
        """Return {seq: (fitness, props)}. Scores fan out to Flash when enabled,
        else fall back to the local rate-limited ThreadPoolExecutor. If a Flash
        generation fails (cold-start/eviction/timeout), degrade to local for the
        rest of the run instead of crashing the GA mid-evolution."""
        if self._flash_enabled:
            logger.debug("_evaluate_population: dispatching %d seqs via Flash.", len(population))
            try:
                scores = self._score_via_flash(population)
            except Exception as e:
                logger.warning("Flash scoring failed (%s); falling back to local "
                               "threadpool for remaining generations.", e)
                self._flash_enabled = False
                scores = self._score_via_threads(population)
        else:
            logger.debug("_evaluate_population: dispatching %d seqs via local threads.", len(population))
            scores = self._score_via_threads(population)
        results = {}
        for seq, sc in zip(population, scores):
            # props are cheap local motif counts; only the score is fanned out.
            results[seq] = (max(0.0, sc), self.oracle.evaluate_sequence_properties(seq))
        return results

    def _score_via_threads(self, population):
        max_w = API_MAX_WORKERS if self._uses_api else LOCAL_MAX_WORKERS
        logger.info("_score_via_threads: pop=%d, max_workers=%s, api=%s",
                    len(population), max_w, self._uses_api)
        scores = [0.0] * len(population)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
            future_to_idx = {}
            for i, seq in enumerate(population):
                future_to_idx[executor.submit(self.oracle.score, seq)] = i
                if self._uses_api and i % API_MAX_WORKERS == (API_MAX_WORKERS - 1):
                    time.sleep(API_CALL_DELAY)
            for future in concurrent.futures.as_completed(future_to_idx):
                scores[future_to_idx[future]] = future.result()
        logger.info("Thread scoring complete: %d scores, mean=%.3f", len(scores),
                    sum(scores) / max(len(scores), 1))
        return scores

    def _score_via_flash(self, population):
        logger.info("_score_via_flash: pop=%d", len(population))
        import asyncio
        import math
        from flash_fitness import score_endpoint, MAX_WORKERS

        # Size batches so the population spreads across the whole fleet (the 3 -> N headline):
        # pop=100, MAX_WORKERS=50 -> chunk=2 -> 50 jobs -> ~50 live workers.
        chunk = max(1, math.ceil(len(population) / MAX_WORKERS))
        logger.info("Flash batch sizing: chunk=%d -> ~%d jobs across fleet (MAX_WORKERS=%d)",
                    chunk, math.ceil(len(population) / chunk), MAX_WORKERS)

        if self._flash_metrics is None:
            from flash_metrics import FanoutMetrics
            self._flash_metrics = FanoutMetrics(phase="ga-fitness", resource="cpu5c-4-8")
        metrics = self._flash_metrics

        async def _run():
            batches = [population[i:i + chunk] for i in range(0, len(population), chunk)]
            logger.debug("_run: dispatching %d batch(es) to the Flash fleet...", len(batches))

            async def _await(batch):
                rec = metrics.start()
                try:
                    # runpod-flash decorator endpoints are AWAITED directly (returns the result
                    # dict) — NOT .run()/job.wait(), which is the client/image-mode API and
                    # raises AttributeError on a decorator endpoint.
                    out = await asyncio.wait_for(
                        score_endpoint({"sequences": batch, "mode": "Local"}), timeout=300)
                    result = out["scores"]
                except Exception:
                    # Flag the failed batch so the metrics stay self-consistent
                    # (n_jobs == n_ok + n_failed), then re-raise: _evaluate_population
                    # catches it and re-scores the whole population on the local threadpool.
                    metrics.done(rec, ok=False)
                    raise
                metrics.done(rec, ok=True)
                logger.debug("_await: batch complete, got %d score(s)", len(result))
                return result

            batch_scores = await asyncio.gather(*(_await(b) for b in batches))
            scores = []
            for s in batch_scores:
                scores.extend(s)
            logger.debug("_run: all batches done, total %d scores", len(scores))
            return scores

        return asyncio.run(_run())

    def _initialize_population(self):
        logger.info("Initializing population: size=%d, genome_length=%d", POPULATION_SIZE, GENOME_LENGTH)
        bases = ['A', 'C', 'G', 'T']
        pop = [''.join(random.choices(bases, k=GENOME_LENGTH)) for _ in range(POPULATION_SIZE)]
        logger.info("Population initialized (%d sequences).", len(pop))
        return pop

    def mutate(self, sequence):
        seq_list = list(sequence)
        num_mutations = max(1, int(len(sequence) * self.mutation_rate))
        indices = random.sample(range(len(sequence)), num_mutations)
        for i in indices:
            seq_list[i] = random.choice(['A', 'C', 'G', 'T'])
        result = "".join(seq_list)
        logger.debug("mutate: %d mutation(s) applied (rate=%.3f)", num_mutations, self.mutation_rate)
        return result

    def crossover(self, parent1, parent2):
        pt1 = random.randint(1, GENOME_LENGTH - 2)
        pt2 = random.randint(pt1 + 1, GENOME_LENGTH - 1)
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        logger.debug("crossover: points=(%d, %d)", pt1, pt2)
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
        
        logger.info("Opening evolution log: %s", LOG_FILE)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Generation", "Best_Fitness", "Tumor_Score", "Normal_Score", "Heat_Score", "Mutation_Rate"])

        for gen in range(GENERATIONS):
            results = self._evaluate_population(self.population)

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
            
            logger.debug("Gen %02d: best_fit=%.2f, mut_rate=%.3f, tumor=%.1f, normal=%.1f, heat=%.1f",
                         gen, best_fit, self.mutation_rate,
                         best_props["tumor_score"], best_props["normal_score"], best_props["heat_score"])
            if gen % 10 == 0 or gen == GENERATIONS - 1:
                logger.info("Gen %02d: Best Fitness=%.2f | MutRate=%.3f", gen, best_fit, self.mutation_rate)
                logger.info("       [Scores] Tumor: %.1f | Normal: %.1f | Heat: %.1f",
                            best_props["tumor_score"], best_props["normal_score"], best_props["heat_score"])
                logger.debug("       [Seq Start] %s...", best_seq[:30])

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

        if self._flash_metrics is not None:
            self._flash_metrics.save()
        return ranked_pop[0], self.history

if __name__ == "__main__":
    oracle = AlphaGenomeOracle()
    optimizer = GeneticOptimizer(oracle)
    best_promoter, history = optimizer.run()
    logger.info("--- Optimization Complete ---")
    logger.info(f"Final Synthetic Promoter: {best_promoter}")
    logger.info(f"Evolution log saved to {LOG_FILE}")
