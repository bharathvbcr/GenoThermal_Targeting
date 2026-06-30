"""
Geno-Thermal Targeting: Master Pipeline (script port of Geno_Thermal_Master.ipynb)

Faithful, headless .py port of the narrative master notebook: same phases, same
AlphaFold-Server-based flow, same charts (saved to outputs/figures/ instead of
plt.show()/display()). This is the original end-to-end narrative pipeline and is
intentionally simpler than run_pipeline.py, which orchestrates the newer
Boltz-2 + RunPod Flash production pipeline as separate phase scripts.

Run: python geno_thermal_master.py
"""
import sys
import os
import json
import random
import re
import logging

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
HARD_MODE_DIR = os.path.join(PROJECT_ROOT, "hard_mode")
for _p in (PROJECT_ROOT, HARD_MODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("geno_thermal_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GenoThermalMaster")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.rcParams["figure.facecolor"] = "white"


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Figure saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Phase 1 — Genomic Discovery (AlphaGenome)
# ---------------------------------------------------------------------------
def phase1_genomic_discovery():
    from alphagenome_utils import AlphaGenomeClient

    logger.info("--- Phase 1: Genomic Discovery (AlphaGenome) ---")

    fasta_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "sample_gene.fasta")
    target_gene = "EGFR"
    normal_seq = "ATCGGCTAACGGCTAACTTAGCCTAGCGTTAACCGGTTATATCGGCTAA"

    ag_client = AlphaGenomeClient(force_local=True)
    mutated_seq = ag_client.parse_fasta(fasta_path)

    logger.info("Normal  seq: %s", normal_seq)
    logger.info("Patient seq: %s", mutated_seq)
    logger.info("Identical?   %s", normal_seq == mutated_seq)

    phase1_result = ag_client.get_expression_score(
        gene_id=target_gene,
        normal_seq=normal_seq,
        mutated_seq=mutated_seq,
    )
    logger.debug(json.dumps(phase1_result, indent=2))

    # --- Decision Gate ---
    preds = phase1_result["predictions"]
    classification = preds["classification"]
    confidence = preds["confidence"]
    epi = preds["epigenetic_profile"]

    logger.info("=" * 50)
    logger.info("TARGET GENE:    %s", target_gene)
    logger.info("CLASSIFICATION: %s", classification)
    logger.info("CONFIDENCE:     %s", confidence)
    logger.info("Normal Score:   %s", preds['normal_score'])
    logger.info("Mutated Score:  %s", preds['mutated_score'])
    logger.info("Epigenetics:    H3K27ac=%s, H3K4me1=%s, H3K27me3=%s",
                epi['H3K27ac'], epi['H3K4me1'], epi['H3K27me3'])
    logger.info("=" * 50)

    if classification == "SUPER_ENHANCER":
        logger.info(">>> DECISION: Proceed with %s as therapy target.", target_gene)
        selected_target = target_gene
    else:
        logger.info(">>> DECISION: No actionable variant found. Pipeline halted.")
        selected_target = None

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    scores = [preds["normal_score"], preds["mutated_score"]]
    colors = ["steelblue", "crimson"]
    axes[0].bar(["Normal", "Mutated (Patient)"], scores, color=colors, edgecolor="black")
    axes[0].set_ylabel("Predicted Expression Score")
    axes[0].set_title(f"{target_gene} Expression Prediction")
    axes[0].axhline(y=50, linestyle="--", color="gray", alpha=0.5, label="Threshold")
    axes[0].legend()

    marks = list(epi.keys())
    vals = [1.0 if v == "High" else 0.2 for v in epi.values()]
    mark_colors = ["#e74c3c" if v == "High" else "#95a5a6" for v in epi.values()]
    axes[1].barh(marks, vals, color=mark_colors, edgecolor="black")
    axes[1].set_xlim(0, 1.2)
    axes[1].set_xlabel("Level")
    axes[1].set_title("Epigenetic Profile (Mutated)")

    savefig("target_expression.png")

    report_path = os.path.join(REPORTS_DIR, "target_report.json")
    with open(report_path, "w") as f:
        json.dump(phase1_result, f, indent=2)
    logger.info("Phase 1 report saved to %s", report_path)

    return {
        "ag_client": ag_client,
        "phase1_result": phase1_result,
        "target_gene": target_gene,
        "classification": classification,
        "selected_target": selected_target,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Ligand Engineering (AlphaFold) & Phase 3 — Structure Visualization
# ---------------------------------------------------------------------------
def phase2_ligand_engineering_and_phase3_viz(target_gene):
    from alphafold_utils import AlphaFoldClient

    logger.info("--- Phase 2: Ligand Engineering (AlphaFold) ---")

    egfr_seq = (
        "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALN"
        "TVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWR"
        "DIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQC"
        "AAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCV"
        "RACGADSYEMEEDGVRKC"
    )

    candidates_path = os.path.join(PROJECT_ROOT, "data", "sample_data", "candidates.csv")
    candidates_df = pd.read_csv(candidates_path)
    logger.info("Loaded %d candidates:\n%s", len(candidates_df), candidates_df.to_string())

    af_client = AlphaFoldClient()

    # --- AlphaFold Server batch JSON export (one job per candidate) ---
    peptide_candidates = [
        {"name": row["name"].replace(" ", "_"), "seq": row["seq"]}
        for _, row in candidates_df.iterrows()
    ]
    batch_files = af_client.create_batch_jobs(target_seq=egfr_seq, peptide_candidates=peptide_candidates)
    logger.info("Total batch files created: %d", len(batch_files))
    for i, path in enumerate(batch_files):
        logger.info("  [%d] %s", i + 1, path)
    logger.info("Upload these files to https://alphafoldserver.com/ -> 'Upload JSON', "
                "download result ZIPs into %s/, then re-run.", af_client.results_dir)

    # --- Parse any already-downloaded results and join back onto the candidate table ---
    # job_name follows create_batch_jobs' "dock_<sanitized candidate name>" convention.
    parsed = af_client.parse_all_results()
    by_job_name = {r["job_name"]: r for r in parsed if r.get("job_name")}

    docking_results = []
    for _, row in candidates_df.iterrows():
        expected_job_name = f"dock_{af_client._sanitize_name(row['name'])}"
        match = by_job_name.get(expected_job_name, {})
        plddt = match.get("plddt_score")
        docking_results.append({
            **row.to_dict(),
            "structure_path": match.get("structure_path"),
            "plddt_score": plddt,
            "pae_score": match.get("pae_score"),
            "classification": AlphaFoldClient.classify_binding(plddt) if plddt is not None else "PENDING",
        })

    results_df = pd.DataFrame(docking_results)
    has_results = results_df["plddt_score"].notna().any()

    if has_results:
        results_df = results_df.sort_values("plddt_score", ascending=False).reset_index(drop=True)
        logger.info("\n%s", results_df[["name", "seq", "plddt_score", "pae_score", "classification"]].to_string())
    else:
        logger.info("=" * 70)
        logger.info("NO ALPHAFOLD RESULTS YET")
        logger.info("=" * 70)
        logger.info("\n%s", results_df[["name", "seq", "classification"]].to_string())

    if has_results:
        best = results_df.iloc[0]
        logger.info("Best Candidate : %s", best['name'])
        logger.info("Sequence       : %s", best['seq'])
        logger.info("pLDDT          : %.1f", best['plddt_score'])
        logger.info("Classification : %s", best['classification'])
        best_peptide_name = best["name"]
        lib_path = os.path.join(REPORTS_DIR, "candidate_library.csv")
        results_df.to_csv(lib_path, index=False)
        logger.info("Candidate library saved to %s", lib_path)
    else:
        best_peptide_name = "GE11_EGF_Mimic"
        best = {"name": best_peptide_name, "seq": "YHWYGYTPQNVI",
                "plddt_score": None, "pae_score": None,
                "classification": "PENDING"}
        logger.info("Using default best peptide: %s", best_peptide_name)
        logger.info("Re-run after downloading AlphaFold results for real scores.")

    # --- Phase 3: Structure Analysis & Visualization ---
    logger.info("--- Phase 3: Structure Analysis & Visualization ---")

    if has_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        names = results_df["name"].tolist()
        plddt_vals = results_df["plddt_score"].tolist()
        pae_vals = results_df["pae_score"].tolist()
        color_map = {"STRONG_BINDER": "#2ecc71", "MODERATE_BINDER": "#f1c40f",
                     "WEAK_BINDER": "#f39c12", "NON_BINDER": "#e74c3c"}
        bar_colors = [color_map.get(c, "gray") for c in results_df["classification"]]
        axes[0].barh(names, plddt_vals, color=bar_colors, edgecolor="black")
        axes[0].axvline(x=80, color="green", linestyle="--", alpha=0.7, label="High-confidence")
        axes[0].set_xlabel("pLDDT")
        axes[0].set_title("Binding Confidence (pLDDT)")
        axes[0].legend(fontsize=8)
        axes[1].barh(names, pae_vals, color=bar_colors, edgecolor="black")
        axes[1].axvline(x=5, color="green", linestyle="--", alpha=0.7, label="Low-error")
        axes[1].set_xlabel("PAE (Å)")
        axes[1].set_title("Predicted Aligned Error")
        axes[1].legend(fontsize=8)
        class_counts = results_df["classification"].value_counts()
        axes[2].bar(class_counts.index, class_counts.values,
                    color=[color_map.get(c, "gray") for c in class_counts.index], edgecolor="black")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Binding Classification Summary")
        savefig("docking_comparison.png")
    else:
        logger.info("Skipping Phase 3 comparison chart — no AlphaFold results yet.")

    if has_results and best.get("structure_path"):
        try:
            import py3Dmol
            struct_path = os.path.join(PROJECT_ROOT, best["structure_path"])
            with open(struct_path, "r") as f:
                struct_data = f.read()
            fmt = "cif" if struct_path.endswith(".cif") else "pdb"
            view = py3Dmol.view(width=600, height=400)
            view.addModel(struct_data, fmt)
            view.setStyle({"cartoon": {"color": "spectrum"}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.3, "color": "white"})
            view.zoomTo()
            html_path = os.path.join(FIGURES_DIR, "best_docking_3d.html")
            view.write_html(html_path)
            logger.info("3D view of %s docked to %s saved to %s", best['name'], target_gene, html_path)
        except ImportError:
            logger.warning("py3Dmol not installed. Install with: pip install py3Dmol")
        except Exception as e:
            logger.warning("3D viewer error: %s", e)
    else:
        logger.info("No structure file available for 3D viewing.")

    return {
        "af_client": af_client,
        "has_results": has_results,
        "results_df": results_df,
        "best": best,
        "best_peptide_name": best_peptide_name,
        "egfr_seq": egfr_seq,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Evolutionary Promoter Design (GA)
# ---------------------------------------------------------------------------
def phase4_evolutionary_promoter():
    from evolver import AlphaGenomeOracle, GeneticOptimizer, calculate_fitness

    logger.info("--- Phase 4: Evolutionary Promoter Design (GA) ---")

    oracle = AlphaGenomeOracle(mode="Local")
    optimizer = GeneticOptimizer(oracle)
    best_promoter, evolution_history = optimizer.run()

    final_fit, final_props = calculate_fitness(best_promoter, oracle)
    logger.info("=" * 60)
    logger.info("EVOLVED SYNTHETIC PROMOTER")
    logger.info("=" * 60)
    logger.info("Length : %d bp", len(best_promoter))
    logger.info("Seq    : %s...", best_promoter[:80])
    logger.info("Fitness       : %.2f", final_fit)
    logger.info("Tumor Score   : %.1f", final_props['tumor_score'])
    logger.info("Normal Score  : %.1f  (lower is better)", final_props['normal_score'])
    logger.info("Heat Score    : %.1f", final_props['heat_score'])
    logger.info("GC Penalty    : %.1f", final_props['gc_penalty'])
    logger.info("Motif counts  : Tumor=%s, Normal=%s, Heat=%s", *final_props['raw_counts'])

    # --- Convergence Analysis ---
    gens = [h[0] for h in evolution_history]
    fits = [h[1] for h in evolution_history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, fits, color="darkorange", linewidth=2)
    ax.fill_between(gens, 0, fits, alpha=0.15, color="orange")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Genetic Algorithm Convergence — Synthetic Promoter Evolution")
    ax.grid(True, alpha=0.3)
    savefig("promoter_convergence.png")

    # --- Sequence Composition & Motif Distribution ---
    gc_count = best_promoter.count("G") + best_promoter.count("C")
    gc_frac = gc_count / len(best_promoter)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bases = ["A", "T", "G", "C"]
    counts = [best_promoter.count(b) for b in bases]
    axes[0].bar(bases, counts, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"], edgecolor="black")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Nucleotide Composition (GC = {gc_frac:.1%})")

    motif_labels = ["ARE (Tumor)", "ERG (Tumor)", "MYC (Tumor)",
                    "TATA (Normal)", "CAAT (Normal)", "CpG (Normal)",
                    "HSE-1 (Heat)", "HSE-2 (Heat)"]
    all_motifs = oracle.tumor_motifs + oracle.normal_motifs + oracle.heat_motifs
    motif_counts = [len(re.findall(m, best_promoter)) for m in all_motifs]
    motif_colors = ["#2ecc71"] * 3 + ["#e74c3c"] * 3 + ["#f39c12"] * 2
    axes[1].barh(motif_labels, motif_counts, color=motif_colors, edgecolor="black")
    axes[1].set_xlabel("Occurrences in Evolved Promoter")
    axes[1].set_title("Motif Distribution")
    savefig("promoter_composition.png")

    return {"best_promoter": best_promoter, "final_fit": final_fit}


# ---------------------------------------------------------------------------
# Phase 5 — Thermo-Switch Protein Design
# ---------------------------------------------------------------------------
def phase5_thermo_switch():
    from thermo_fold import ThermoSwitchOptimizer, BASE_SCAFFOLD

    logger.info("--- Phase 5: Thermo-Switch Protein Design ---")

    designer = ThermoSwitchOptimizer(BASE_SCAFFOLD)
    best_switch_seq, best_tm = designer.run()

    logger.info("=" * 60)
    logger.info("DESIGNED THERMO-SWITCH PROTEIN")
    logger.info("=" * 60)
    logger.info("Sequence     : %s", best_switch_seq)
    logger.info("Length       : %d aa", len(best_switch_seq))
    logger.info("Predicted Tm : %.1f °C", best_tm)

    phys = designer.oracle
    plddt_37 = phys.predict_plddt(best_switch_seq, 37.0)
    plddt_43 = phys.predict_plddt(best_switch_seq, 43.0)
    logger.info("pLDDT @ 37°C : %.1f", plddt_37)
    logger.info("pLDDT @ 43°C : %.1f", plddt_43)
    logger.info("Switch gap   : %.1f (larger = sharper switch)", plddt_37 - plddt_43)

    # Reuses ThermoSwitchOptimizer.plot_melting_curve() -> outputs/figures/thermo_profile.png
    designer.plot_melting_curve(best_switch_seq, best_tm)

    return {"best_switch_seq": best_switch_seq, "best_tm": best_tm}


# ---------------------------------------------------------------------------
# Phase 6 — Nanoparticle Surface Topology (Monte Carlo)
# ---------------------------------------------------------------------------
def phase6_nano_topology():
    from nano_topology import NanoTopologySim, GRID_SIZE, LIGAND, PEG, EMPTY

    logger.info("--- Phase 6: Nanoparticle Surface Topology (Monte Carlo) ---")

    nano_sim = NanoTopologySim()
    nano_sim.run_annealing()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.colors.ListedColormap(["white", "crimson", "dodgerblue"])
    axes[0].imshow(nano_sim.grid, cmap=cmap, interpolation="nearest")
    legend_elements = [
        mpatches.Patch(facecolor="crimson", label="Ligand (GE11)"),
        mpatches.Patch(facecolor="dodgerblue", label="Stealth (PEG)"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Empty"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    axes[0].set_title("Optimized Nanoparticle Surface")
    axes[0].axis("off")

    total = GRID_SIZE * GRID_SIZE
    lig_count = int(np.sum(nano_sim.grid == LIGAND))
    peg_count = int(np.sum(nano_sim.grid == PEG))
    empty_count = int(np.sum(nano_sim.grid == EMPTY))

    labels = ["Ligand", "PEG", "Empty"]
    sizes = [lig_count, peg_count, empty_count]
    colors_pie = ["crimson", "dodgerblue", "lightgray"]
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "black"})
    axes[1].set_title("Surface Composition")

    savefig("nano_surface_coverage.png")
    logger.info("Coverage — Ligand: %.1f%% | PEG: %.1f%% | Empty: %.1f%%",
                100 * lig_count / total, 100 * peg_count / total, 100 * empty_count / total)

    return {"lig_count": lig_count, "peg_count": peg_count, "total": total}


# ---------------------------------------------------------------------------
# Phase 7 — Biological Circuit Integration (AND Gate)
# ---------------------------------------------------------------------------
def phase7_bio_circuit(best_promoter, best_switch_seq):
    from bio_circuit import BioCircuitSimulator

    logger.info("--- Phase 7: Biological Circuit Integration (AND Gate) ---")

    circuit = BioCircuitSimulator(promoter_seq=best_promoter, switch_seq=best_switch_seq)
    temps_circuit, heatmap_data = circuit.run_simulation()

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".1f", cmap="RdYlGn_r",
        xticklabels=[f"{t:.0f}" for t in temps_circuit],
        yticklabels=["NORMAL", "TUMOR"],
        linewidths=0.5, ax=ax,
    )
    ax.set_title("AND Logic Gate — Kill Switch Activation (%)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Cell Context")
    savefig("circuit_heatmap_narrative.png")

    normal_signal = heatmap_data[0, :]
    tumor_signal = heatmap_data[1, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps_circuit, tumor_signal, color="red", linewidth=2, label="TUMOR (target)")
    ax.plot(temps_circuit, normal_signal, color="blue", linewidth=2, label="NORMAL (bystander)")
    ax.fill_between(temps_circuit, normal_signal, tumor_signal,
                    alpha=0.1, color="red", label="Therapeutic Window")
    ax.axvline(x=37, color="green", linestyle="--", alpha=0.5, label="Body Temp")
    ax.axvline(x=43, color="orange", linestyle="--", alpha=0.5, label="Hyperthermia")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Kill Signal (%)")
    ax.set_title("Therapeutic Window — Tumor vs. Normal Tissue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("therapeutic_window.png")

    return {"temps_circuit": temps_circuit, "tumor_signal": tumor_signal, "normal_signal": normal_signal}


# ---------------------------------------------------------------------------
# Phase 8 — RL-Driven Sequence Design (PPO Agent)
# ---------------------------------------------------------------------------
def phase8_rl_design():
    logger.info("--- Phase 8: RL-Driven Sequence Design (PPO Agent) ---")

    try:
        from rl_gene_designer import PromoterDesignEnv
    except ImportError as e:
        logger.warning("Skipping Phase 8 entirely: %s", e)
        logger.warning("Install with: pip install gymnasium stable-baselines3")
        return {"rl_designed_dna": None}

    test_env = PromoterDesignEnv(target_length=20, mode='Local')
    obs, _ = test_env.reset()
    done = False
    while not done:
        action = test_env.action_space.sample()
        obs, reward, done, _, _ = test_env.step(action)
    logger.info("Random agent sanity check: DNA=%s Reward=%.2f",
                test_env._indices_to_string(obs), reward)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        class ProgressCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)

            def _on_step(self) -> bool:
                if self.n_calls % 1000 == 0:
                    logger.info("  Step %d: Training...", self.n_calls)
                return True

        rl_seq_length = 50
        rl_timesteps = 10_000
        logger.info("Training PPO agent (length=%dbp, timesteps=%d)...", rl_seq_length, rl_timesteps)
        env = DummyVecEnv([lambda: PromoterDesignEnv(target_length=rl_seq_length, mode="Local")])
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                     n_steps=2048, batch_size=64, gamma=0.99,
                     device="cpu",
                     tensorboard_log="./outputs/ppo_gene_tensorboard/")
        model.learn(total_timesteps=rl_timesteps, callback=ProgressCallback())
        logger.info("Training complete.")

        gen_env = PromoterDesignEnv(target_length=rl_seq_length, mode='Local')
        obs, _ = gen_env.reset()
        done = False
        reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, reward, done, _, _ = gen_env.step(action)

        rl_designed_dna = gen_env._indices_to_string(gen_env.sequence)
        logger.info("RL-Designed DNA (%dbp): %s", rl_seq_length, rl_designed_dna)
        logger.info("Fitness: %.2f", reward)
        return {"rl_designed_dna": rl_designed_dna}

    except ImportError as e:
        logger.warning("Skipping PPO training: %s", e)
        logger.warning("Install with: pip install stable-baselines3 gymnasium")
        return {"rl_designed_dna": None}


# ---------------------------------------------------------------------------
# Phase 9 — Physics Verification (OpenMM / CUDA)
# ---------------------------------------------------------------------------
def phase9_physics_verification(phase2_3):
    logger.info("--- Phase 9: Physics Verification (OpenMM / CUDA) ---")

    pdb_file = os.path.join(PROJECT_ROOT, "outputs", "simulated_pdbs", "egfr_ge11_complex.pdb")
    if phase2_3["has_results"] and phase2_3["best"].get("structure_path"):
        af_struct = os.path.join(PROJECT_ROOT, phase2_3["best"]["structure_path"])
        if os.path.exists(af_struct):
            pdb_file = af_struct
            logger.info("Using AlphaFold structure: %s", pdb_file)

    try:
        import openmm as mm
        from openmm import app, unit

        fidelity_mode = "production"  # "demo"=10ps, "production"=1ns, "high_fidelity"=5ns
        sim_map = {"demo": 5000, "production": 500000, "high_fidelity": 2500000}
        simulation_steps = sim_map.get(fidelity_mode, 5000)

        def fix_structure(pdb_path):
            try:
                from pdbfixer import PDBFixer
                logger.info("  PDBFixer: repairing %s...", os.path.basename(pdb_path))
                fixer = PDBFixer(filename=pdb_path)
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)
                return fixer.topology, fixer.positions
            except ImportError:
                logger.warning("  PDBFixer not installed — loading PDB directly.")
                pdb = app.PDBFile(pdb_path)
                return pdb.topology, pdb.positions

        def setup_simulation(pdb_path, temperature_kelvin):
            logger.info("Setting up simulation at %sK...", temperature_kelvin)
            topology, positions = fix_structure(pdb_path)
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            modeller = app.Modeller(topology, positions)
            modeller.addSolvent(forcefield, padding=1.0 * unit.nanometers)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometers,
                constraints=app.HBonds,
            )
            integrator = mm.LangevinMiddleIntegrator(
                temperature_kelvin * unit.kelvin,
                1.0 / unit.picosecond,
                0.002 * unit.picoseconds,
            )
            try:
                platform = mm.Platform.getPlatformByName('CUDA')
                props = {'Precision': 'mixed'}
                logger.info("  Platform: CUDA (GPU)")
            except Exception:
                platform = mm.Platform.getPlatformByName('CPU')
                props = {}
                logger.info("  Platform: CPU (slow)")
            simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
            simulation.context.setPositions(modeller.positions)
            return simulation

        def compute_aligned_rmsd(p1, p2):
            p1_c = p1 - np.mean(p1, axis=0)
            p2_c = p2 - np.mean(p2, axis=0)
            C = np.dot(p1_c.T, p2_c)
            V, S, W_t = np.linalg.svd(C)
            if (np.linalg.det(V) * np.linalg.det(W_t)) < 0.0:
                S[-1] = -S[-1]
                V[:, -1] = -V[:, -1]
            U = np.dot(V, W_t)
            p1_aligned = np.dot(p1_c, U)
            diff = p1_aligned - p2_c
            return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        def run_md(simulation, label):
            logger.info("[%s] Minimizing...", label)
            simulation.minimizeEnergy()
            logger.info("[%s] Equilibrating (100 steps)...", label)
            simulation.step(100)
            initial_pos = simulation.context.getState(getPositions=True).getPositions()
            logger.info("[%s] Production (%d steps)...", label, simulation_steps)
            simulation.step(simulation_steps)
            final_pos = simulation.context.getState(getPositions=True).getPositions()
            p1 = np.array(initial_pos.value_in_unit(unit.nanometers))
            p2 = np.array(final_pos.value_in_unit(unit.nanometers))
            rmsd = compute_aligned_rmsd(p1, p2)
            logger.info("[%s] Aligned RMSD: %.4f nm", label, rmsd)
            return rmsd

        if os.path.exists(pdb_file):
            sim_37 = setup_simulation(pdb_file, 310.15)
            rmsd_37 = run_md(sim_37, "37C")
            sim_43 = setup_simulation(pdb_file, 316.15)
            rmsd_43 = run_md(sim_43, "43C")

            logger.info("RMSD @ 37°C: %.4f nm", rmsd_37)
            logger.info("RMSD @ 43°C: %.4f nm", rmsd_43)

            if rmsd_37 < 0.5 and rmsd_43 > rmsd_37 * 1.2:
                logger.info("PASS: Protein behaves as a thermal switch.")
            else:
                logger.info("FAIL: Insufficient thermal switching.")
        else:
            logger.warning("PDB file not found: %s", pdb_file)
            logger.warning("Run Phase 2 first to generate structure files.")

    except ImportError:
        logger.warning("OpenMM not installed. Skipping physics verification.")
        logger.warning("Install with: conda install -c conda-forge openmm")


# ---------------------------------------------------------------------------
# Final Summary Report
# ---------------------------------------------------------------------------
def print_summary(phase1, phase2_3, phase4, phase5, phase6, phase7, phase8):
    logger.info("=" * 70)
    logger.info("        GENO-THERMAL TARGETING — PIPELINE SUMMARY")
    logger.info("=" * 70)

    logger.info("[Phase 1] Genomic Discovery (AlphaGenome)")
    logger.info("  Target Gene      : %s", phase1["selected_target"])
    logger.info("  Classification   : %s", phase1["classification"])
    logger.info("  Compute          : %s", phase1["ag_client"]._mode)

    logger.info("[Phase 2] Ligand Engineering (AlphaFold Server)")
    logger.info("  Best Peptide     : %s", phase2_3["best_peptide_name"])
    if phase2_3["has_results"]:
        logger.info("  pLDDT            : %s", phase2_3["best"]['plddt_score'])
        logger.info("  Classification   : %s", phase2_3["best"]['classification'])
    else:
        logger.info("  Status           : PENDING")

    logger.info("[Phase 4] Evolved Promoter (GA)")
    logger.info("  Length            : %d bp", len(phase4["best_promoter"]))
    logger.info("  Fitness           : %.2f", phase4["final_fit"])

    logger.info("[Phase 5] Thermo-Switch")
    logger.info("  Designed Tm       : %.1f °C", phase5["best_tm"])

    logger.info("[Phase 6] Nanoparticle Surface")
    logger.info("  Ligand Coverage   : %.1f%%", 100 * phase6["lig_count"] / phase6["total"])
    logger.info("  PEG Coverage      : %.1f%%", 100 * phase6["peg_count"] / phase6["total"])

    logger.info("[Phase 7] Circuit")
    temps_circuit = phase7["temps_circuit"]
    tumor_signal = phase7["tumor_signal"]
    normal_signal = phase7["normal_signal"]
    idx_37 = int(np.argmin(np.abs(temps_circuit - 37.0)))
    idx_43 = int(np.argmin(np.abs(temps_circuit - 43.0)))
    logger.info("  Tumor  @ 43°C    : %.1f%% kill", tumor_signal[idx_43])
    logger.info("  Normal @ 37°C    : %.1f%% kill", normal_signal[idx_37])
    selectivity = tumor_signal[idx_43] / max(normal_signal[idx_37], 0.01)
    logger.info("  Selectivity       : %.1fx", selectivity)

    if phase8["rl_designed_dna"] is not None:
        logger.info("[Phase 8] RL Agent")
        logger.info("  Designed DNA      : %s...", phase8["rl_designed_dna"][:40])

    logger.info("=" * 70)
    logger.info("Pipeline complete.")


def main():
    phase1 = phase1_genomic_discovery()
    phase2_3 = phase2_ligand_engineering_and_phase3_viz(phase1["target_gene"])
    phase4 = phase4_evolutionary_promoter()
    phase5 = phase5_thermo_switch()
    phase6 = phase6_nano_topology()
    phase7 = phase7_bio_circuit(phase4["best_promoter"], phase5["best_switch_seq"])
    phase8 = phase8_rl_design()
    phase9_physics_verification(phase2_3)
    print_summary(phase1, phase2_3, phase4, phase5, phase6, phase7, phase8)


if __name__ == "__main__":
    main()
