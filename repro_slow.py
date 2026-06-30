
import os
import json
import sys
import logging
from alphagenome_utils import AlphaGenomeClient

logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("repro_slow.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ReproSlow")

if "ALPHAGENOME_API_KEY" in os.environ:
    logger.info("API Key found: %s...", os.environ["ALPHAGENOME_API_KEY"][:5])
else:
    logger.warning("No API Key in env; will use local fallback.")

PROJECT_ROOT = os.getcwd()
FASTA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_data", "sample_gene.fasta")
TARGET_GENE = "EGFR"

NORMAL_SEQ = "ATCGGCTAACGGCTAACTTAGCCTAGCGTTAACCGGTTATATCGGCTAA"

logger.info("Initializing AlphaGenomeClient...")
ag_client = AlphaGenomeClient()
logger.info("Client mode: %s", ag_client._mode)

logger.info("Parsing FASTA from %s", FASTA_PATH)
mutated_seq = ag_client.parse_fasta(FASTA_PATH)
logger.info("Mutated seq length: %d", len(mutated_seq))

logger.info("Getting expression score for %s...", TARGET_GENE)
phase1_result = ag_client.get_expression_score(
    gene_id=TARGET_GENE,
    normal_seq=NORMAL_SEQ,
    mutated_seq=mutated_seq,
)

logger.info("Result:\n%s", json.dumps(phase1_result, indent=2))
