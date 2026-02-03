
import os
import json
import sys
from alphagenome_utils import AlphaGenomeClient

# Mock environmental setup
# Explicitly UNSET the key to test fallback first, or SET it if we want to test API path (but we can't if we don't have a key)
if "ALPHAGENOME_API_KEY" in os.environ:
    print(f"API Key found: {os.environ['ALPHAGENOME_API_KEY'][:5]}...")
else:
    print("No API Key in env.")

# --- Configuration ---
# Point to the actual file
PROJECT_ROOT = os.getcwd()
FASTA_PATH = os.path.join(PROJECT_ROOT, "sample_data", "sample_gene.fasta")
TARGET_GENE = "EGFR"

NORMAL_SEQ = "ATCGGCTAACGGCTAACTTAGCCTAGCGTTAACCGGTTATATCGGCTAA"

# --- Initialize AlphaGenome Client ---
print("Initializing client...")
ag_client = AlphaGenomeClient()

print(f"Client mode: {ag_client._mode}")

print("Parsing fasta...")
mutated_seq = ag_client.parse_fasta(FASTA_PATH)
print(f"Mutated seq length: {len(mutated_seq)}")

print("Getting expression score...")
phase1_result = ag_client.get_expression_score(
    gene_id=TARGET_GENE,
    normal_seq=NORMAL_SEQ,
    mutated_seq=mutated_seq,
)

print(json.dumps(phase1_result, indent=2))
