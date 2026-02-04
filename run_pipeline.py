import subprocess
import sys
import os
import time
import logging

# Setup logging for the master pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PipelineMaster")

def run_step(command, description):
    logger.info(f"--- STEP: {description} ---")
    logger.info(f"Executing: {command}")
    
    start_time = time.time()
    try:
        # Use shell=True for complex commands or Windows compatibility with certain invokes
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        logger.info(f"SUCCESS: {description} (Time: {elapsed:.2f}s)")
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"FAILED: {description} after {elapsed:.2f}s")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def check_openmm():
    try:
        import openmm
        return True
    except ImportError:
        return False

def main():
    logger.info("Geno-Thermal Targeting: Master Pipeline Starting")
    
    # 1. Genomic Discovery
    if not run_step("python genomic_discovery.py --target_gene EGFR", 
                    "Phase 1: Genomic Discovery"):
        sys.exit(1)

    # 2. Ligand Design
    if not run_step("python ligand_designer.py --output_csv candidate_library_v2.csv", 
                    "Phase 2: Ligand Engineering"):
        sys.exit(1)
        
    # 3. Thermo-Switch Design
    if not run_step("python hard_mode/thermo_fold.py",
                    "Phase 5: Thermo-Switch Protein Design"):
        sys.exit(1)

    # 4. Nano Topology
    if not run_step("python hard_mode/nano_topology.py",
                    "Phase 6: Nanoparticle Surface Topology"):
        sys.exit(1)

    # 5. Bio Circuit
    if not run_step("python hard_mode/bio_circuit.py",
                    "Phase 7: Biological Circuit Integration"):
        sys.exit(1)

    # 6. Evolutionary Design (Hard Mode)
    if not run_step("python hard_mode/evolver.py", 
                    "Phase 4: Evolutionary Promoter Design"):
        sys.exit(1)

    # 7. RL Training (Hard Mode)
    if not run_step("python hard_mode/ppo_agent.py", 
                    "Phase 8: RL-Driven Sequence Design"):
        sys.exit(1)

    # 8. Physics Verification (Conditional)
    if check_openmm():
        if not run_step("python hard_mode/physics_verify.py", 
                        "Phase 9: Physics Verification (OpenMM)"):
            logger.warning("Physics verification failed.")
    else:
        logger.info("SKIPPING Phase 9: Physics Verification ('openmm' not found)")

    # 9. Visualization
    if not run_step("python visualize_results.py", 
                    "Phase 10: Visualization"):
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info("Artifacts generated. See 'pipeline_master.log' for details.")

if __name__ == "__main__":
    main()
