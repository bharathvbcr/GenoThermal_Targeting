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
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
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
PDB_FILE = "outputs/simulated_pdbs/unknown_complex.pdb"
FORCEFIELD_TYPE = 'amber14-all.xml'
WATER_MODEL = 'amber14/tip3p.xml'
SIMULATION_STEPS = 5000 
REPORT_INTERVAL = 100

def fix_pdb(pdb_file):
    """Uses pdbfixer to fix common topology issues in AlphaFold PDBs."""
    if not PDBFIXER_AVAILABLE:
        logger.debug("fix_pdb: pdbfixer unavailable, returning original file.")
        return pdb_file

    logger.info("fix_pdb: fixing %s", pdb_file)
    fixer = PDBFixer(filename=pdb_file)
    logger.debug("fix_pdb: findMissingResidues...")
    fixer.findMissingResidues()
    logger.debug("fix_pdb: findNonstandardResidues / replaceNonstandardResidues...")
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    logger.debug("fix_pdb: findMissingAtoms / addMissingAtoms...")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    logger.debug("fix_pdb: addMissingHydrogens (pH 7.0)...")
    fixer.addMissingHydrogens(7.0)

    fixed_pdb = pdb_file.replace(".pdb", "_fixed.pdb")
    with open(fixed_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    logger.info("fix_pdb: fixed PDB written to %s", fixed_pdb)
    return fixed_pdb

def setup_simulation(pdb_file, temperature_kelvin):
    logger.info("setup_simulation: %s at %.2f K", pdb_file, temperature_kelvin)

    if PDBFIXER_AVAILABLE:
        logger.debug("setup_simulation: running fix_pdb...")
        pdb_file = fix_pdb(pdb_file)

    logger.debug("setup_simulation: loading PDB from %s", pdb_file)
    pdb = app.PDBFile(pdb_file)
    logger.debug("setup_simulation: creating ForceField (%s, %s)", FORCEFIELD_TYPE, WATER_MODEL)
    forcefield = app.ForceField(FORCEFIELD_TYPE, WATER_MODEL)

    logger.debug("setup_simulation: building Modeller...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if not PDBFIXER_AVAILABLE:
        logger.debug("setup_simulation: addHydrogens (pdbfixer absent)...")
        modeller.addHydrogens(forcefield)

    logger.debug("setup_simulation: addSolvent (padding=1.0 nm)...")
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers)

    logger.debug("setup_simulation: createSystem (PME, HBonds)...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometers,
        constraints=app.HBonds
    )

    logger.debug("setup_simulation: LangevinMiddleIntegrator at %.2f K", temperature_kelvin)
    integrator = mm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0/unit.picosecond,
        0.002*unit.picoseconds
    )

    platform = None
    props = {}
    for platform_name in ['CUDA', 'OpenCL']:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            if platform_name == 'CUDA':
                props = {'Precision': 'mixed'}
            logger.info("setup_simulation: using platform %s (GPU accelerated)", platform_name)
            break
        except Exception as _plat_err:
            logger.debug("setup_simulation: platform %s unavailable (%s), trying next.", platform_name, _plat_err)

    if platform is None:
        platform = mm.Platform.getPlatformByName('CPU')
        logger.warning("setup_simulation: falling back to CPU platform (MD will be slow).")

    logger.debug("setup_simulation: creating Simulation object...")
    simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(modeller.positions)
    logger.info("setup_simulation: simulation ready (%.2f K).", temperature_kelvin)
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
    result = float(np.sqrt((diff * diff).sum() / len(p1)))
    logger.debug("calculate_rmsd: n_atoms=%d, RMSD=%.4f nm", len(p1), result)
    return result

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