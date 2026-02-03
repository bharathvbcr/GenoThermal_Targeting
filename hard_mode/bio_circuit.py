import numpy as np
import matplotlib.pyplot as plt

class BioCircuitSimulator:
    def __init__(self, promoter_seq, switch_seq):
        self.promoter_seq = promoter_seq
        self.switch_seq = switch_seq
        
    def get_promoter_activity(self, context, temp):
        """
        Simulates the synthetic promoter evolved in evolver.py.
        """
        # Logic derived from our Genetic Algorithm results
        # High activity ONLY in Tumor context.
        # Temperature has a minor synergistic effect on HSF1 binding (Heat Shock Factor).
        
        base_activity = 0
        if context == "TUMOR":
            base_activity = 95.0
        elif context == "NORMAL":
            base_activity = 5.0 # Leaky expression
            
        # Thermal boost (HSF1 recruitment)
        if temp >= 42.0:
            base_activity *= 1.2 # 20% boost from heat shock elements
            
        return min(100.0, base_activity)

    def get_switch_state(self, temp):
        """
        Simulates the Thermo-Switch protein from thermo_fold.py.
        Uses a sigmoid transfer function for melting around Tm = 40C.
        """
        # Sigmoid: 1 / (1 + exp(-k * (T - Tm)))
        # k = 1.5 (steepness), Tm = 40.0
        tm = 40.0
        k = 1.5
        percent_active = 100.0 / (1.0 + np.exp(-k * (temp - tm)))
            
        return percent_active

    def run_simulation(self):
        contexts = ["NORMAL", "TUMOR"]
        temps = np.arange(36.0, 46.0, 0.5)
        
        results = []
        
        print(f"--- Bio-Circuit Simulation ---")
        print(f"Promoter: {self.promoter_seq[:10]}...")
        print(f"Protein:  {self.switch_seq[:10]}...")
        print("-" * 60)
        print(f"{ 'Context':<10} | { 'Temp':<6} | { 'Promoter':<10} | { 'Switch':<10} | { 'KILL SIGNAL':<12}")
        print("-" * 60)

        heatmap_data = np.zeros((len(contexts), len(temps)))

        for i, ctx in enumerate(contexts):
            for j, t in enumerate(temps):
                p_act = self.get_promoter_activity(ctx, t)
                s_act = self.get_switch_state(t)
                
                # AND Gate Logic: Both must be high
                # Normalize to 0-1
                kill_signal = (p_act / 100.0) * (s_act / 100.0) * 100.0
                
                heatmap_data[i, j] = kill_signal
                
                # Print key checkpoints
                if t in [37.0, 43.0]:
                    print(f"{ctx:<10} | {t:<6.1f} | {p_act:<10.1f} | {s_act:<10.1f} | {kill_signal:<12.1f}")

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
        print("\nCircuit heatmap saved to 'circuit_heatmap.png'")

if __name__ == "__main__":
    # Sequences from previous steps
    promoter = "TCCGAACCTCCGCCGTTGCCGCCGACCGCCGTCAGCTCGTCCGTGACGAG"
    protein = "MKQLEDKVEELASKNYHLENEVARLLKLVGER"
    
    sim = BioCircuitSimulator(promoter, protein)
    temps, data = sim.run_simulation()
    sim.plot_circuit(temps, data)
