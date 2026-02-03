import pandas as pd
import os
import json

def main():
    print("--- Geno-Thermal Targeting: Project Summary ---\n")

    # 1. Check Target Report
    if os.path.exists("target_report.json"):
        with open("target_report.json", "r") as f:
            target_data = json.load(f)
        print(f"Phase 1 Target: {target_data.get('gene_id')} (Conf: {target_data['predictions']['confidence']})")
        print(f"Classification: {target_data['predictions']['classification']}\n")
    else:
        print("Phase 1: target_report.json NOT FOUND.\n")

    # 2. Check Candidate Library
    if os.path.exists("candidate_library.csv"):
        df = pd.read_csv("candidate_library.csv")
        print(f"Phase 2 Candidates Generated: {len(df)}")
        
        # Best Candidate
        if 'plddt_score' in df.columns:
            best = df.loc[df['plddt_score'].idxmax()]
            print(f"Best Candidate: {best['name']}")
            print(f"  Sequence: {best['seq']}")
            print(f"  pLDDT Score: {best['plddt_score']:.2f}")
            print(f"  Binding Energy: {best['binding_energy_kcal_mol']} kcal/mol")
            print(f"  Model File: {best['structure_path']}")
            
            if best['plddt_score'] > 80:
                print("\nSUCCESS: Identified High-Confidence Binder!")
            else:
                print("\nWARNING: No high-confidence binder found (threshold 80).")
        else:
            print("Error: plddt_score column missing in CSV.")
    else:
        print("Phase 2: candidate_library.csv NOT FOUND.\n")

    # 3. Check Visualization Notebook
    if os.path.exists("structure_viz.ipynb"):
        print("\nPhase 3: structure_viz.ipynb is present.")
    else:
        print("\nPhase 3: structure_viz.ipynb is MISSING.")

if __name__ == "__main__":
    main()
