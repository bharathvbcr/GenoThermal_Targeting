# hard_mode/physics_verify.py

"""
The In-Silico Cell: RL-Driven Design of Thermo-Genetic Circuits
Module 3: The Physics Verification Engine (OpenMM)
"""

import sys
import os
import time
import argparse
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("physics_verify.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhysicsVerify")

try:
    import openmm as mm
    from openmm import app
    from openmm import unit
except ImportError:
    logger.error("OpenMM is not installed. Please install it to run physics simulations.")
    sys.exit(1)

# Check for pdbfixer
try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False
    logger.warning("pdbfixer not installed. Topology issues may occur.")

# Configuration
PDB_FILE = "simulated_pdbs/unknown_complex.pdb"
FORCEFIELD_TYPE = 'amber14-all.xml'
WATER_MODEL = 'amber14/tip3p.xml'
SIMULATION_STEPS = 5000 
REPORT_INTERVAL = 100

def fix_pdb(pdb_file):
    """Uses pdbfixer to fix common topology issues in AlphaFold PDBs."""
    if not PDBFIXER_AVAILABLE:
        return pdb_file
    
    logger.info(f"Fixing PDB: {pdb_file}")
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    fixed_pdb = pdb_file.replace(".pdb", "_fixed.pdb")
    with open(fixed_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return fixed_pdb

def setup_simulation(pdb_file, temperature_kelvin):
    logger.info(f"--- Setting up simulation for {pdb_file} at {temperature_kelvin}K ---")
    
    # Optional: Fix PDB first
    if PDBFIXER_AVAILABLE:
        pdb_file = fix_pdb(pdb_file)

    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField(FORCEFIELD_TYPE, WATER_MODEL)
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if not PDBFIXER_AVAILABLE:
        modeller.addHydrogens(forcefield)
    
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers)
    
    system = forcefield.createSystem(
        modeller.topology, 
        nonbondedMethod=app.PME, 
        nonbondedCutoff=1.0*unit.nanometers, 
        constraints=app.HBonds
    )
    
    integrator = mm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin, 
        1.0/unit.picosecond, 
        0.002*unit.picoseconds
    )
    
    # Try to use GPU Acceleration (CUDA then OpenCL), fallback to CPU
    platform = None
    props = {}
    
    for platform_name in ['CUDA', 'OpenCL']:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            if platform_name == 'CUDA':
                props = {'Precision': 'mixed'}
            logger.info(f"Using Platform: {platform_name} (GPU Accelerated)")
            break
        except Exception:
            continue
            
    if platform is None:
        platform = mm.Platform.getPlatformByName('CPU')
        logger.info("Using Platform: CPU (Warning: Slow for MD)")
        
    simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(modeller.positions)
    
    return simulation

def run_md_protocol(simulation, name):
    logger.info(f"[{name}] Minimizing Energy...")
    simulation.minimizeEnergy()
    logger.info(f"[{name}] Equilibrating (100 steps)...")
    simulation.step(100)
    logger.info(f"[{name}] Running Production MD ({SIMULATION_STEPS} steps)...")
    
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, REPORT_INTERVAL, step=True, 
        potentialEnergy=True, temperature=True, speed=True
    ))
    
    initial_state = simulation.context.getState(getPositions=True)
    initial_positions = initial_state.getPositions()
    
    simulation.step(SIMULATION_STEPS)
    
    final_state = simulation.context.getState(getPositions=True)
    final_positions = final_state.getPositions()
    
    rmsd = calculate_rmsd(initial_positions, final_positions)
    logger.info(f"[{name}] Final RMSD: {rmsd:.4f} nm")
    return rmsd

def calculate_rmsd(pos1, pos2):
    p1 = np.array(pos1.value_in_unit(unit.nanometers))
    p2 = np.array(pos2.value_in_unit(unit.nanometers))
    diff = p1 - p2
    return np.sqrt((diff * diff).sum() / len(p1))

def verify_thermal_switch(pdb_path):
    try:
        sim_37 = setup_simulation(pdb_path, 310.15)
        rmsd_37 = run_md_protocol(sim_37, "BodyTemp_37C")
        
        sim_43 = setup_simulation(pdb_path, 316.15)
        rmsd_43 = run_md_protocol(sim_43, "Hyperthermia_43C")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False
    
    logger.info("\n--- Physics Verification Results ---")
    logger.info(f"RMSD @ 37C: {rmsd_37:.4f} nm")
    logger.info(f"RMSD @ 43C: {rmsd_43:.4f} nm")
    
    stability_pass = rmsd_37 < 0.5 
    switching_pass = rmsd_43 > (rmsd_37 * 1.2)
    
    if stability_pass and switching_pass:
        logger.info("PASS: Protein behaves as a Thermal Switch.")
        return True
    else:
        logger.info("FAIL: Thermal switching criteria not met.")
    return False

if __name__ == "__main__":
    if not os.path.exists(PDB_FILE):
        logger.error(f"File not found: {PDB_FILE}")
        sys.exit(1)
    verify_thermal_switch(PDB_FILE)