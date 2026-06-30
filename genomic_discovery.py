import argparse
import json
import os
import logging
from env_utils import load_dotenv
from alphagenome_utils import AlphaGenomeClient

# Load .env so ALPHAGENOME_API_KEY reaches AlphaGenomeClient when this phase is run
# standalone (run_pipeline.py also loads it, but a no-op double-load is harmless and
# never overrides an already-set var).
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genomic_discovery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GenomicDiscovery")

def main():
    parser = argparse.ArgumentParser(
        description="Geno-Thermal Targeting: Phase 1 - Genomic Discovery"
    )
    parser.add_argument("--input_file", type=str,
                        default="data/sample_data/sample_gene.fasta",
                        help="Path to FASTA/VCF file.")
    parser.add_argument("--target_gene", type=str, default="EGFR",
                        help="Target gene ID (default: EGFR)")
    parser.add_argument("--output_file", type=str,
                        default="outputs/reports/target_report.json",
                        help="Path to save the report.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="AlphaGenome API key (or set ALPHAGENOME_API_KEY env var)")
    args = parser.parse_args()

    logger.info(f"--- Starting Genomic Discovery for {args.target_gene} ---")

    client = AlphaGenomeClient(api_key=args.api_key)

    sequence_data = client.parse_fasta(args.input_file)
    mutated_seq = sequence_data
    logger.info("Mutated sequence loaded: %d bp", len(mutated_seq))

    normal_seq = "ATCGGCTAACGGCTAACTTAGCCTAGCGTTAACCGGTTATATCGGCTAA"
    logger.debug("Normal baseline: %d bp", len(normal_seq))

    if mutated_seq == normal_seq:
        logger.info("Input sequence matches baseline (Normal — no mutations detected).")
    else:
        logger.info("Input sequence differs from baseline (%d bp vs %d bp).",
                    len(mutated_seq), len(normal_seq))

    logger.info("Querying for expression prediction...")
    result = client.get_expression_score(
        gene_id=args.target_gene,
        normal_seq=normal_seq,
        mutated_seq=mutated_seq,
    )

    logger.info("Analysis Result received.")
    logger.debug(json.dumps(result, indent=2))

    # Persist provenance so downstream consumers (and the HTML report) can tell whether
    # these numbers came from the real AlphaGenome API or the synthetic local fallback,
    # instead of having to guess from the values.
    result.setdefault("data_source", getattr(client, "_mode", "UNKNOWN"))
    logger.info("AlphaGenome data source: %s", result["data_source"])

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Report saved to: {args.output_file}")

if __name__ == "__main__":
    main()
