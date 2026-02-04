import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_evolution_log(log_file="evolution_log.csv"):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Run 'python hard_mode/evolver.py' first.")
        return

    print(f"Plotting {log_file}...")
    df = pd.read_csv(log_file)
    
    # Setup style
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Fitness (Left Y-Axis)
    color_fit = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness', color=color_fit)
    sns.lineplot(data=df, x='Generation', y='Best_Fitness', ax=ax1, color=color_fit, linewidth=2.5, label='Fitness')
    ax1.tick_params(axis='y', labelcolor=color_fit)

    # Plot Mutation Rate (Right Y-Axis)
    if 'Mutation_Rate' in df.columns:
        ax3 = ax1.twinx()
        color_mut = 'tab:purple'
        # Offset the spine to make room for component scores
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.set_ylabel('Mutation Rate', color=color_mut)
        sns.lineplot(data=df, x='Generation', y='Mutation_Rate', ax=ax3, color=color_mut, linewidth=1.5, linestyle='--', label='Mutation Rate')
        ax3.tick_params(axis='y', labelcolor=color_mut)

    # Plot Component Scores (Secondary Y-Axis - Inner Right)
    ax2 = ax1.twinx()
    
    # Melt for component plotting
    df_melt = df.melt(id_vars=['Generation'], value_vars=['Tumor_Score', 'Normal_Score', 'Heat_Score'], 
                      var_name='Component', value_name='Score')
    
    sns.lineplot(data=df_melt, x='Generation', y='Score', hue='Component', ax=ax2, 
                 palette={'Tumor_Score': 'green', 'Normal_Score': 'red', 'Heat_Score': 'orange'},
                 style='Component', markers=True, dashes=False, alpha=0.6)
    
    ax2.set_ylabel('Component Scores')
    
    plt.title('Evolution of Synthetic Promoter Fitness')
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    if 'Mutation_Rate' in df.columns:
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    else:
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    
    output_file = "evolution_trajectory.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_evolution_log(sys.argv[1])
    else:
        plot_evolution_log()