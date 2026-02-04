"""
AlphaFold Server Client — Submit protein-peptide folding jobs to AlphaFold Server.
"""

import json
import os
import hashlib
import zipfile
import glob
import numpy as np
import logging
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlphaFoldClient")

class AlphaFoldClient:
    """
    Client for interacting with AlphaFold Server.
    """

    def __init__(self, output_dir="alphafold_jobs", results_dir="alphafold_results"):
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.pdb_dir = "predicted_structures"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.pdb_dir, exist_ok=True)
        logger.info(f"AlphaFold Client initialized. Jobs: {self.output_dir}, Results: {self.results_dir}")

    def _sanitize_name(self, name):
        import re
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        clean = re.sub(r"__+", "_", clean)
        return clean.strip("_")

    def create_docking_job(self, job_name, target_seq, peptide_seq,
                           target_copies=1, peptide_copies=1):
        sanitized_name = self._sanitize_name(job_name)
        job = {
            "name": sanitized_name,
            "modelSeeds": [],
            "sequences": [
                {"proteinChain": {"sequence": target_seq, "count": target_copies}},
                {"proteinChain": {"sequence": peptide_seq, "count": peptide_copies}},
            ],
            "dialect": "alphafoldserver",
            "version": 1,
        }
        filepath = os.path.join(self.output_dir, f"job_{sanitized_name.lower()}.json")
        with open(filepath, "w") as f:
            json.dump(job, f, indent=2)
        logger.info(f"Job created: {filepath}. Upload to https://alphafoldserver.com/")
        return filepath

    def create_batch_jobs(self, target_seq, peptide_candidates):
        all_job_files = []
        batch_size = 100
        for i in range(0, len(peptide_candidates), batch_size):
            batch = peptide_candidates[i:i + batch_size]
            jobs = []
            for cand in batch:
                sanitized_cand_name = self._sanitize_name(cand['name'])
                jobs.append({
                    "name": f"dock_{sanitized_cand_name}",
                    "modelSeeds": [],
                    "sequences": [
                        {"proteinChain": {"sequence": target_seq, "count": 1}},
                        {"proteinChain": {"sequence": cand["seq"], "count": 1}},
                    ],
                    "dialect": "alphafoldserver",
                    "version": 1,
                })
            batch_num = i // batch_size + 1
            filename = f"batch_docking_jobs_{batch_num}.json" if len(peptide_candidates) > batch_size else "batch_docking_jobs.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(jobs, f, indent=2)
            all_job_files.append(filepath)
            logger.info(f"Batch package {batch_num} created: {filepath} ({len(jobs)} jobs)")
        return all_job_files

    def _parse_result_dir(self, extract_dir):
        """
        Parses all models in the directory and returns the one with the highest pLDDT.
        """
        cif_files = sorted(glob.glob(os.path.join(extract_dir, "*_model_*.cif")))
        if not cif_files:
            return {}

        best_plddt = -1
        best_model = {}

        for cif in cif_files:
            model_idx = cif.split("_model_")[-1].replace(".cif", "")
            summary_file = os.path.join(extract_dir, f"{os.path.basename(cif).replace(f'_model_{model_idx}.cif', f'_summary_confidences_{model_idx}.json')}")
            
            # Fallback to finding any summary if naming is weird
            if not os.path.exists(summary_file):
                summaries = glob.glob(os.path.join(extract_dir, f"*_summary_confidences_{model_idx}.json"))
                if summaries: summary_file = summaries[0]

            plddt = 0
            pae = 0
            if os.path.exists(summary_file):
                with open(summary_file) as f:
                    conf = json.load(f)
                plddt = conf.get("ptm", conf.get("iptm", 0.0)) * 100
                pae = conf.get("ranking_score", 0.0)
            
            if plddt > best_plddt:
                best_plddt = plddt
                best_model = {
                    "structure_path": cif,
                    "plddt_score": plddt,
                    "pae_score": pae,
                    "model_index": model_idx
                }

        if best_model:
            dest = os.path.join(self.pdb_dir, os.path.basename(best_model["structure_path"]))
            shutil.copy2(best_model["structure_path"], dest)
            best_model["structure_path"] = dest
            
            job_files = glob.glob(os.path.join(extract_dir, "*_job_request.json"))
            if job_files:
                with open(job_files[0]) as f:
                    job_data = json.load(f)
                if isinstance(job_data, list): job_data = job_data[0]
                best_model["job_name"] = job_data.get("name", "unknown")
        
        return best_model

    def parse_result_zip(self, zip_path):
        extract_dir = os.path.join(self.results_dir, os.path.splitext(os.path.basename(zip_path))[0])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return self._parse_result_dir(extract_dir)

    def parse_all_results(self):
        results = []
        for zp in sorted(glob.glob(os.path.join(self.results_dir, "*.zip"))):
            logger.info(f"Parsing ZIP: {os.path.basename(zp)}")
            results.append(self.parse_result_zip(zp))
        for entry in sorted(os.listdir(self.results_dir)):
            entry_path = os.path.join(self.results_dir, entry)
            if os.path.isdir(entry_path):
                if glob.glob(os.path.join(entry_path, "*_model_*.cif")):
                    logger.info(f"Parsing dir: {entry}")
                    results.append(self._parse_result_dir(entry_path))
        return [r for r in results if r]

    @staticmethod
    def classify_binding(plddt):
        if plddt >= 80: return "STRONG_BINDER"
        elif plddt >= 60: return "MODERATE_BINDER"
        elif plddt >= 40: return "WEAK_BINDER"
        return "NON_BINDER"