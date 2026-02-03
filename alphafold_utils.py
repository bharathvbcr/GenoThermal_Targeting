"""
AlphaFold Server Client — Submit protein-peptide folding jobs to AlphaFold Server.

AlphaFold Server (alphafoldserver.com) provides AlphaFold 3 predictions for
protein complexes including peptides, DNA, RNA, and ligands.

Workflow:
  1. Build a JSON job definition for the receptor-peptide complex.
  2. Save to disk (for manual upload) or auto-submit if API access is available.
  3. Poll for results / parse downloaded result ZIP files.
  4. Extract pLDDT, PAE, and PDB structure.

JSON spec: https://github.com/google-deepmind/alphafold/blob/main/server/README.md
Server:    https://alphafoldserver.com/
"""

import json
import os
import hashlib
import zipfile
import glob
import numpy as np


class AlphaFoldClient:
    """
    Client for interacting with AlphaFold Server.

    Two modes of operation:
      - JOB_EXPORT: Generates JSON job files for upload to alphafoldserver.com
      - RESULT_PARSE: Parses downloaded result ZIPs (PDB, scores, confidence)

    AlphaFold Server does not currently expose a public REST API for automated
    job submission. Jobs are submitted via the web UI or JSON upload.
    This client automates job generation and result parsing.
    """

    def __init__(self, output_dir="alphafold_jobs", results_dir="alphafold_results"):
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.pdb_dir = "predicted_structures"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.pdb_dir, exist_ok=True)
        print(f"AlphaFold Client initialized.")
        print(f"  Job export dir  : {self.output_dir}/")
        print(f"  Results dir     : {self.results_dir}/")
        print(f"  Structure dir   : {self.pdb_dir}/")

    def _sanitize_name(self, name):
        """Removes parentheses and other non-shell-friendly characters."""
        import re
        # Replace spaces, parens, and special chars with underscores
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        # Collapse multiple underscores
        clean = re.sub(r"__+", "_", clean)
        return clean.strip("_")

    # ------------------------------------------------------------------
    # Job generation — creates JSON files for AlphaFold Server upload
    # ------------------------------------------------------------------
    def create_docking_job(self, job_name, target_seq, peptide_seq,
                           target_copies=1, peptide_copies=1):
        """
        Generate an AlphaFold Server JSON job for receptor-peptide docking.

        Returns:
            Path to the generated JSON file.
        """
        sanitized_name = self._sanitize_name(job_name)
        job = {
            "name": sanitized_name,
            "modelSeeds": [],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": target_seq,
                        "count": target_copies,
                    }
                },
                {
                    "proteinChain": {
                        "sequence": peptide_seq,
                        "count": peptide_copies,
                    }
                },
            ],
            "dialect": "alphafoldserver",
            "version": 1,
        }

        filename = f"job_{sanitized_name.lower()}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(job, f, indent=2)

        print(f"Job created: {filepath}")
        print(f"  Upload this file to https://alphafoldserver.com/ -> 'Upload JSON'")
        return filepath

    def create_batch_jobs(self, target_seq, peptide_candidates):
        """
        Generate batch JSON files containing multiple docking jobs.
        Each file contains a maximum of 100 jobs to comply with AlphaFold Server limits.

        Args:
            target_seq:         Receptor amino acid sequence.
            peptide_candidates: List of dicts with keys 'name' and 'seq'.

        Returns:
            List of paths to the generated batch JSON files.
        """
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
            print(f"Batch job package {batch_num} created: {filepath} ({len(jobs)} jobs)")

        print(f"Total: {len(all_job_files)} files created. Upload to https://alphafoldserver.com/ -> 'Upload JSON'")
        return all_job_files

    # ------------------------------------------------------------------
    # Result parsing — extract scores from ZIP files or directories
    # ------------------------------------------------------------------
    def _parse_result_dir(self, extract_dir):
        """
        Parse an AlphaFold Server result directory (already extracted).

        The directory typically contains:
          - *_model_0.cif  (structure in mmCIF format)
          - *_summary_confidences_0.json (pLDDT, PAE summary)
          - *_full_data_0.json (detailed per-residue data)
          - fold_*_job_request.json (original job parameters)

        Returns:
            dict with keys: structure_path, plddt_score, pae_score,
                            confidence_data, job_name
        """
        result = {}

        # Find key files
        cif_files = glob.glob(os.path.join(extract_dir, "*_model_*.cif"))
        confidence_files = glob.glob(os.path.join(extract_dir, "*_summary_confidences_*.json"))
        full_data_files = glob.glob(os.path.join(extract_dir, "*_full_data_*.json"))
        job_files = glob.glob(os.path.join(extract_dir, "*_job_request.json"))

        # Structure file
        if cif_files:
            import shutil
            dest = os.path.join(self.pdb_dir, os.path.basename(cif_files[0]))
            shutil.copy2(cif_files[0], dest)
            result["structure_path"] = dest

        # Confidence scores (from summary)
        if confidence_files:
            with open(sorted(confidence_files)[0]) as f:
                conf = json.load(f)
            result["plddt_score"] = conf.get("ptm", conf.get("iptm", 0.0)) * 100
            result["pae_score"] = conf.get("ranking_score", 0.0)
            result["confidence_data"] = conf

        # Full data (per-residue pLDDT — overrides summary if available)
        if full_data_files:
            with open(sorted(full_data_files)[0]) as f:
                full = json.load(f)
            if "atom_plddts" in full:
                plddts = full["atom_plddts"]
                result["plddt_score"] = float(np.mean(plddts))
            if "pae" in full:
                pae_matrix = np.array(full["pae"])
                result["pae_score"] = float(np.mean(pae_matrix))

        # Job name
        if job_files:
            with open(job_files[0]) as f:
                job_data = json.load(f)
            # Job request can be a list (batch) or dict (single)
            if isinstance(job_data, list):
                job_data = job_data[0]
            result["job_name"] = job_data.get("name", "unknown")

        return result

    def parse_result_zip(self, zip_path):
        """Parse an AlphaFold Server result ZIP file."""
        extract_dir = os.path.join(
            self.results_dir,
            os.path.splitext(os.path.basename(zip_path))[0]
        )
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return self._parse_result_dir(extract_dir)

    def parse_all_results(self):
        """
        Parse all result ZIPs and extracted directories in the results folder.

        Returns:
            List of result dicts, one per job.
        """
        results = []

        # Parse ZIP files
        for zp in sorted(glob.glob(os.path.join(self.results_dir, "*.zip"))):
            print(f"Parsing ZIP: {os.path.basename(zp)}")
            results.append(self.parse_result_zip(zp))

        # Parse already-extracted directories (contain *_model_*.cif)
        for entry in sorted(os.listdir(self.results_dir)):
            entry_path = os.path.join(self.results_dir, entry)
            if os.path.isdir(entry_path):
                cifs = glob.glob(os.path.join(entry_path, "*_model_*.cif"))
                if cifs:
                    print(f"Parsing dir: {entry}")
                    results.append(self._parse_result_dir(entry_path))

        if not results:
            print(f"No results found in {self.results_dir}/")
            print("Download from alphafoldserver.com and place ZIPs or extracted folders here.")

        return results

    # ------------------------------------------------------------------
    # Classify binding quality from scores
    # ------------------------------------------------------------------
    @staticmethod
    def classify_binding(plddt, pae=None, binding_energy=None):
        """Classify a docking result based on confidence metrics."""
        if plddt >= 80:
            return "STRONG_BINDER"
        elif plddt >= 60:
            return "MODERATE_BINDER"
        elif plddt >= 40:
            return "WEAK_BINDER"
        return "NON_BINDER"

    # ------------------------------------------------------------------
    # Convenience: simulate_docking (backward-compatible interface)
    # ------------------------------------------------------------------
    def _find_result_for_peptide(self, peptide_seq):
        """
        Search results directory for a result matching the given peptide.

        Checks both ZIP files and extracted directories by scanning the
        job_request.json for a matching peptide sequence.
        """
        # Check ZIPs
        for zp in glob.glob(os.path.join(self.results_dir, "*.zip")):
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    for name in zf.namelist():
                        if "job_request" in name:
                            with zf.open(name) as f:
                                job = json.loads(f.read())
                            if isinstance(job, list):
                                job = job[0]
                            for seq_entry in job.get("sequences", []):
                                chain = seq_entry.get("proteinChain", {})
                                if chain.get("sequence") == peptide_seq:
                                    return self.parse_result_zip(zp)
            except Exception:
                continue

        # Check extracted directories
        for entry in os.listdir(self.results_dir):
            entry_path = os.path.join(self.results_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            for jf in glob.glob(os.path.join(entry_path, "*_job_request.json")):
                try:
                    with open(jf) as f:
                        job = json.load(f)
                    if isinstance(job, list):
                        job = job[0]
                    for seq_entry in job.get("sequences", []):
                        chain = seq_entry.get("proteinChain", {})
                        if chain.get("sequence") == peptide_seq:
                            return self._parse_result_dir(entry_path)
                except Exception:
                    continue

        return None

    def simulate_docking(self, target_seq, peptide_seq):
        """
        Backward-compatible interface that generates a job JSON and
        checks if results are already available.

        Searches for results by peptide sequence match (handles
        both ZIP files and extracted directories from AlphaFold Server).
        """
        seq_hash = hashlib.md5(
            (target_seq[:20] + peptide_seq).encode()
        ).hexdigest()[:8]
        job_name = f"dock_{seq_hash}"

        # Search for results matching this peptide sequence
        result = self._find_result_for_peptide(peptide_seq)
        if result and result.get("plddt_score") is not None:
            result["peptide_seq"] = peptide_seq
            result["classification"] = self.classify_binding(
                result.get("plddt_score", 0)
            )
            return result

        # No results yet — generate job file
        job_path = self.create_docking_job(
            job_name=job_name,
            target_seq=target_seq,
            peptide_seq=peptide_seq,
        )

        return {
            "peptide_seq": peptide_seq,
            "plddt_score": None,
            "pae_score": None,
            "binding_energy_kcal_mol": None,
            "structure_path": None,
            "classification": "PENDING",
            "job_file": job_path,
            "status": "Job JSON created. Upload to alphafoldserver.com, "
                      "download results ZIP, place in "
                      f"'{self.results_dir}/', then re-run.",
        }
