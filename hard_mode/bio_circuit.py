import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bio_circuit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BioCircuit")

class BioCircuitSimulator:
    def __init__(self, promoter_seq, switch_seq):
        logger.info("BioCircuitSimulator init: promoter_len=%d, switch_len=%d",
                    len(promoter_seq), len(switch_seq))
        self.promoter_seq = promoter_seq
        self.switch_seq = switch_seq
        
    def get_promoter_activity(self, context, temp):
        base_activity = 95.0 if context == "TUMOR" else 5.0
        heat_boost = temp >= 42.0
        if heat_boost:
            base_activity *= 1.2
        result = min(100.0, base_activity)
        logger.debug("get_promoter_activity: ctx=%s, T=%.1f, heat_boost=%s -> %.1f%%",
                     context, temp, heat_boost, result)
        return result

    def get_switch_state(self, temp):
        tm = 40.0
        k = 1.5
        percent_active = 100.0 / (1.0 + np.exp(-k * (temp - tm)))
        logger.debug("get_switch_state: T=%.1f°C -> %.1f%% active (Tm=%.1f, k=%.1f)",
                     temp, percent_active, tm, k)
        return percent_active

    def run_simulation(self):
        contexts = ["NORMAL", "TUMOR"]
        temps = np.arange(36.0, 46.0, 0.5)
        logger.info("--- Bio-Circuit Simulation --- contexts=%s, temp_range=[%.1f, %.1f]",
                    contexts, float(temps[0]), float(temps[-1]))
        heatmap_data = np.zeros((len(contexts), len(temps)))
        for i, ctx in enumerate(contexts):
            for j, t in enumerate(temps):
                p_act = self.get_promoter_activity(ctx, t)
                s_act = self.get_switch_state(t)
                kill_signal = (p_act / 100.0) * (s_act / 100.0) * 100.0
                heatmap_data[i, j] = kill_signal
                if t in [37.0, 43.0]:
                    logger.info(f"Ctx: {ctx:<7} | T: {t:<4.1f} | P: {p_act:<5.1f} | S: {s_act:<5.1f} | KILL: {kill_signal:<5.1f}")
        logger.info("Bio-circuit simulation complete. Heatmap shape: %s", heatmap_data.shape)
        return temps, heatmap_data

    def plot_circuit(self, temps, data):
        logger.info("Plotting circuit heatmap (%d contexts x %d temp points).", data.shape[0], data.shape[1])
        import seaborn as sns
        plt.figure(figsize=(10, 5))
        sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn_r",
                    xticklabels=[f"{t:.1f}" for t in temps], 
                    yticklabels=["NORMAL", "TUMOR"])
        plt.title("Logic Gate: Kill Switch Activation Level")
        plt.xlabel("Temperature (\u00b0C)")
        plt.ylabel("Cell Context")
        plt.tight_layout()
        os.makedirs("outputs/figures", exist_ok=True)
        plt.savefig("outputs/figures/circuit_heatmap.png")
        logger.info("Circuit heatmap saved to 'outputs/figures/circuit_heatmap.png'")

if __name__ == "__main__":
    promoter = "TCCGAACCTCCGCCGTTGCCGCCGACCGCCGTCAGCTCGTCCGTGACGAG"
    protein = "MKQLEDKVEELASKNYHLENEVARLLKLVGER"
    sim = BioCircuitSimulator(promoter, protein)
    temps, data = sim.run_simulation()
    sim.plot_circuit(temps, data)