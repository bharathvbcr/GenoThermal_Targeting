import argparse
import json
import os
from alphagenome_utils import AlphaGenomeClient

def main():
    parser = argparse.ArgumentParser(
        description="Geno-Thermal Targeting: Phase 1 - Genomic Discovery"
    )
    parser.add_argument("--input_file", type=str,
                        default="sample_data/sample_gene.fasta",
                        help="Path to FASTA/VCF file.")
    parser.add_argument("--target_gene", type=str, default="EGFR",
                        help="Target gene ID (default: EGFR)")
    parser.add_argument("--output_file", type=str,
                        default="target_report.json",
                        help="Path to save the report.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="AlphaGenome API key (or set ALPHAGENOME_API_KEY env var)")
    args = parser.parse_args()

    print(f"--- Starting Genomic Discovery for {args.target_gene} ---")

    client = AlphaGenomeClient(api_key=args.api_key)

    sequence_data = client.parse_fasta(args.input_file)
    mutated_seq = sequence_data

    # Baseline normal sequence
    normal_seq = "ATCGGCTAACGGCTAACTTAGCCTAGCGTTAACCGGTTATATCGGCTAA"

    if mutated_seq == normal_seq:
        print("Note: Input sequence matches baseline (Normal).")

    print("Querying for expression prediction...")
    result = client.get_expression_score(
        gene_id=args.target_gene,
        normal_seq=normal_seq,
        mutated_seq=mutated_seq,
    )

    print("\n--- Analysis Result ---")
    print(json.dumps(result, indent=2))

    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to: {args.output_file}")

if __name__ == "__main__":
    main()
