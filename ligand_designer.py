import argparse
import pandas as pd
import os
import logging
from alphafold_utils import AlphaFoldClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ligand_designer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LigandDesigner")

def main():
    parser = argparse.ArgumentParser(
        description="Geno-Thermal Targeting: Phase 2 - Ligand Engineering"
    )
    parser.add_argument("--target_seq", type=str,
                        help="Amino acid sequence of the target receptor.")
    parser.add_argument("--candidates_file", type=str,
                        default="sample_data/candidates.csv",
                        help="CSV with 'name' and 'seq' columns.")
    parser.add_argument("--output_csv", type=str,
                        default="candidate_library.csv",
                        help="Path to save results CSV.")
    args = parser.parse_args()

    # Default EGFR sequence
    target_seq = args.target_seq
    if not target_seq:
        target_seq = (
            "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALN"
            "TVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWR"
            "DIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQC"
            "AAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCV"
            "RACGADSYEMEEDGVRKC"
        )
        logger.info(f"Using default EGFR target sequence ({len(target_seq)} aa)")

    # Load candidates
    try:
        candidates_df = pd.read_csv(args.candidates_file)
        if 'name' not in candidates_df.columns or 'seq' not in candidates_df.columns:
            raise ValueError("CSV must contain 'name' and 'seq' columns.")
        logger.info(f"Loaded {len(candidates_df)} candidates from {args.candidates_file}")
    except Exception as e:
        logger.warning(f"Error reading candidates: {e}. Using default candidates.")
        candidates_df = pd.DataFrame([
            {"name": "GE11 (EGF Mimic)", "seq": "YHWYGYTPQNVI"},
            {"name": "RGD (Integrin binder)", "seq": "ACDCRGDCFC"},
            {"name": "Poly-Alanine (Neg Control)", "seq": "AAAAAAAAAA"},
        ])

    af_client = AlphaFoldClient()

    # Generate batch job JSON for AlphaFold Server
    candidates_list = candidates_df.to_dict("records")
    batch_paths = af_client.create_batch_jobs(target_seq, candidates_list)
    for path in batch_paths:
        logger.info(f"Batch job created: {path}")

    # Check for any already-downloaded results
    logger.info("Parsing results from local cache...")
    results = af_client.parse_all_results()

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        logger.info(f"Parsed results saved to: {args.output_csv}")
        if "plddt_score" in results_df.columns:
            best = results_df.loc[results_df["plddt_score"].idxmax()]
            logger.info(f"Best candidate found: {best.get('job_name', 'N/A')} (pLDDT: {best['plddt_score']:.1f})")
    else:
        logger.info("No results yet in the results directory.")
        logger.info(f"Upload batch job JSON files to alphafoldserver.com, download result ZIPs into '{af_client.results_dir}/', then re-run.")

if __name__ == "__main__":
    main()
