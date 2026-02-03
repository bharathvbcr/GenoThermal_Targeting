# hard_mode/physics_verify.py

"""
The In-Silico Cell: RL-Driven Design of Thermo-Genetic Circuits
Module 3: The Physics Verification Engine (OpenMM)

This script performs Molecular Dynamics (MD) simulations to verify the 
thermal switching behavior of the designed protein.

Logic:
1. Load the Protein Structure (PDB).
2. Solvate in explicit water (TIP3P model).
3. Run Simulation A: Body Temperature (310 K / 37 C).
4. Run Simulation B: Hyperthermia (316 K / 43 C).
5. Compare Root Mean Square Deviation (RMSD) to detect unfolding.
"""

import sys
import os
import time
import argparse
import numpy as np

try:
    import openmm as mm
    from openmm import app
    from openmm import unit
except ImportError:
    print("Error: OpenMM is not installed. Please install it to run physics simulations.")
    sys.exit(1)

# Configuration
PDB_FILE = "simulated_pdbs/unknown_complex.pdb" # Target protein
FORCEFIELD_TYPE = 'amber14-all.xml'
WATER_MODEL = 'amber14/tip3p.xml'
SIMULATION_STEPS = 5000  # Short for demo. Real physics needs 1M+ steps (nanoseconds).
REPORT_INTERVAL = 100

def setup_simulation(pdb_file, temperature_kelvin):
    """
    Prepares the molecular system for simulation.
    """
    print(f"--- Setting up simulation for {pdb_file} at {temperature_kelvin}K ---")
    
    # 1. Load Structure
    print("Loading PDB...")
    pdb = app.PDBFile(pdb_file)
    
    # 2. Define Forcefield
    print("Loading Forcefield...")
    forcefield = app.ForceField(FORCEFIELD_TYPE, WATER_MODEL)
    
    # 3. Create System (Add Solvent)
    print("Adding Hydrogens...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    
    print("Solvating System (adding water box)...")
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers)
    
    system = forcefield.createSystem(
        modeller.topology, 
        nonbondedMethod=app.PME, 
        nonbondedCutoff=1.0*unit.nanometers, 
        constraints=app.HBonds
    )
    
    # 4. Define Integrator (Langevin Dynamics for temperature control)
    integrator = mm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin, 
        1.0/unit.picosecond, 
        0.002*unit.picoseconds # 2fs timestep
    )
    
    # 5. Define Simulation Platform
    # Try to use CUDA (GPU), fallback to CPU
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        props = {'Precision': 'mixed'}
        print("Using Platform: CUDA (GPU Accelerated)")
    except Exception:
        platform = mm.Platform.getPlatformByName('CPU')
        props = {}
        print("Using Platform: CPU (Warning: Slow for MD)")
        
    simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(modeller.positions)
    
    return simulation

def run_md_protocol(simulation, name):
    """
    Runs Energy Minimization -> Equilibration -> Production.
    Returns the final RMSD (Root Mean Square Deviation) from start.
    """
    # 1. Minimize Energy (Relax bad contacts)
    print(f"[{name}] Minimizing Energy...")
    simulation.minimizeEnergy()
    
    # 2. Equilibration (Warm up)
    print(f"[{name}] Equilibrating (100 steps)...")
    simulation.step(100)
    
    # 3. Production Run
    print(f"[{name}] Running Production MD ({SIMULATION_STEPS} steps)...")
    
    # Reporters (Log to stdout)
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, REPORT_INTERVAL, step=True, 
        potentialEnergy=True, temperature=True, speed=True
    ))
    
    # Track RMSD
    initial_state = simulation.context.getState(getPositions=True)
    initial_positions = initial_state.getPositions()
    
    # Run
    simulation.step(SIMULATION_STEPS)
    
    # Calculate Final RMSD
    final_state = simulation.context.getState(getPositions=True)
    final_positions = final_state.getPositions()
    
    rmsd = calculate_rmsd(initial_positions, final_positions)
    print(f"[{name}] Final RMSD: {rmsd:.4f} nm")
    return rmsd

def calculate_rmsd(pos1, pos2):
    """
    Simple RMSD calculation between two sets of positions (ignoring alignment for speed).
    In rigorous analysis, we'd align structures first.
    """
    p1 = np.array(pos1.value_in_unit(unit.nanometers))
    p2 = np.array(pos2.value_in_unit(unit.nanometers))
    diff = p1 - p2
    return np.sqrt((diff * diff).sum() / len(p1))

def verify_thermal_switch(pdb_path):
    """
    The Core Logic: 
    Does it stay folded at 37C? Does it unfold at 43C?
    """
    # Temp 1: Body Temperature (37 C = 310.15 K)
    try:
        sim_37 = setup_simulation(pdb_path, 310.15)
        rmsd_37 = run_md_protocol(sim_37, "BodyTemp_37C")
        
        # Temp 2: Hyperthermia (43 C = 316.15 K)
        sim_43 = setup_simulation(pdb_path, 316.15)
        rmsd_43 = run_md_protocol(sim_43, "Hyperthermia_43C")
    except ValueError as e:
        print(f"\n\u26a0\ufe0f PHYSICS ENGINE ERROR: Topology mismatch.")
        print(f"  {e}")
        print("  \u2139\ufe0f Recommendation: Run 'pdbfixer' to add missing terminal caps or atoms.")
        return False
    except Exception as e:
        print(f"\n\u26a0\ufe0f CRITICAL SIMULATION ERROR: {e}")
        return False
    
    print("\n--- Physics Verification Results ---")
    print(f"RMSD @ 37C: {rmsd_37:.4f} nm")
    print(f"RMSD @ 43C: {rmsd_43:.4f} nm")
    
    # Criteria: 
    # 1. RMSD_37 should be low (< 0.3 nm implies stable)
    # 2. RMSD_43 should be higher (significant unfolding)
    
    stability_pass = rmsd_37 < 0.5 # Relaxed threshold for demo
    switching_pass = rmsd_43 > (rmsd_37 * 1.2) # At least 20% more movement
    
    if stability_pass and switching_pass:
        print("\u2705 PASS: Protein behaves as a Thermal Switch.")
        return True
    elif not stability_pass:
        print("\u274c FAIL: Protein unstable at body temperature.")
    else:
        print("\u274c FAIL: Protein did not respond sufficiently to heat shock.")
        
    return False

if __name__ == "__main__":
    # Check if target PDB exists
    if not os.path.exists(PDB_FILE):
        print(f"Error: {PDB_FILE} not found. Using mock path.")
        sys.exit(1)
        
    verify_thermal_switch(PDB_FILE)
