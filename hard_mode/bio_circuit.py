import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bio_circuit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BioCircuit")

class BioCircuitSimulator:
    def __init__(self, promoter_seq, switch_seq):
        self.promoter_seq = promoter_seq
        self.switch_seq = switch_seq
        
    def get_promoter_activity(self, context, temp):
        base_activity = 95.0 if context == "TUMOR" else 5.0
        if temp >= 42.0: base_activity *= 1.2
        return min(100.0, base_activity)

    def get_switch_state(self, temp):
        tm = 40.0
        k = 1.5
        percent_active = 100.0 / (1.0 + np.exp(-k * (temp - tm)))
        return percent_active

    def run_simulation(self):
        contexts = ["NORMAL", "TUMOR"]
        temps = np.arange(36.0, 46.0, 0.5)
        logger.info(f"--- Bio-Circuit Simulation ---")
        heatmap_data = np.zeros((len(contexts), len(temps)))
        for i, ctx in enumerate(contexts):
            for j, t in enumerate(temps):
                p_act = self.get_promoter_activity(ctx, t)
                s_act = self.get_switch_state(t)
                kill_signal = (p_act / 100.0) * (s_act / 100.0) * 100.0
                heatmap_data[i, j] = kill_signal
                if t in [37.0, 43.0]:
                    logger.info(f"Ctx: {ctx:<7} | T: {t:<4.1f} | P: {p_act:<5.1f} | S: {s_act:<5.1f} | KILL: {kill_signal:<5.1f}")
        return temps, heatmap_data

    def plot_circuit(self, temps, data):
        import seaborn as sns
        plt.figure(figsize=(10, 5))
        sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn_r",
                    xticklabels=[f"{t:.1f}" for t in temps], 
                    yticklabels=["NORMAL", "TUMOR"])
        plt.title("Logic Gate: Kill Switch Activation Level")
        plt.xlabel("Temperature (\u00b0C)")
        plt.ylabel("Cell Context")
        plt.tight_layout()
        plt.savefig("circuit_heatmap.png")
        logger.info("Circuit heatmap saved to 'circuit_heatmap.png'")

if __name__ == "__main__":
    promoter = "TCCGAACCTCCGCCGTTGCCGCCGACCGCCGTCAGCTCGTCCGTGACGAG"
    protein = "MKQLEDKVEELASKNYHLENEVARLLKLVGER"
    sim = BioCircuitSimulator(promoter, protein)
    temps, data = sim.run_simulation()
    sim.plot_circuit(temps, data)