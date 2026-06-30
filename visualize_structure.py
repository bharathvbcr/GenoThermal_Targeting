import argparse
import glob
import json
import os
import logging

import py3Dmol
from Bio.PDB import MMCIFParser, PDBParser

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualize_structure.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StructureViewer")

CHAIN_COLORS = ["skyblue", "orange", "limegreen", "magenta", "gold"]

SEARCH_DIRS = [
    "outputs/predicted_structures",
    "outputs/alphafold_results",
    "outputs/simulated_pdbs",
]


def resolve_structure_path(name_or_path):
    """Resolve a bare job name to a structure file under the known output dirs."""
    if os.path.isfile(name_or_path):
        return name_or_path

    for base in SEARCH_DIRS:
        for ext in ("cif", "pdb"):
            matches = sorted(glob.glob(os.path.join(base, "**", f"*{name_or_path}*model_0.{ext}"), recursive=True))
            if matches:
                return matches[0]
            matches = sorted(glob.glob(os.path.join(base, "**", f"*{name_or_path}*.{ext}"), recursive=True))
            if matches:
                return matches[0]

    raise FileNotFoundError(f"No structure file found matching '{name_or_path}' under {SEARCH_DIRS}")


def load_summary_confidences(structure_path):
    """If an AlphaFold summary_confidences JSON sits next to the structure, load it."""
    base = structure_path
    for suffix in ("_model_0.cif", "_model_0.pdb"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        base, _ = os.path.splitext(base)

    summary_path = base.replace("model_0", "summary_confidences_0")
    if summary_path == base:
        summary_path = f"{base}_summary_confidences_0.json"

    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        logger.info("Loaded confidence summary from %s", summary_path)
        return data

    logger.debug("No summary_confidences file found alongside %s", structure_path)
    return None


def detect_chain_ids(structure_path, fmt):
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("structure", structure_path)
    return sorted(chain.id for chain in structure[0])


def render_structure(structure_path, style="cartoon", spin=False):
    ext = os.path.splitext(structure_path)[1].lstrip(".").lower()
    fmt = "cif" if ext == "cif" else "pdb"

    with open(structure_path) as f:
        data = f.read()

    logger.info("Rendering %s (%s, %d bytes)", structure_path, fmt, len(data))

    view = py3Dmol.view(width=900, height=650)
    view.addModel(data, fmt)

    chain_ids = detect_chain_ids(structure_path, fmt)
    logger.info("Detected chains: %s", chain_ids)

    if len(chain_ids) <= 1:
        view.setStyle({}, {style: {"color": "spectrum"}})
    else:
        for i, chain in enumerate(chain_ids):
            color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
            view.setStyle({"chain": chain}, {style: {"color": color}})

    view.zoomTo()
    if spin:
        view.spin(True)

    return view, chain_ids


def main():
    parser = argparse.ArgumentParser(
        description="Geno-Thermal Targeting: 3D structure viewer (py3Dmol)"
    )
    parser.add_argument("structure", type=str,
                        help="Path to a .cif/.pdb file, or a job-name fragment to search for "
                             "under outputs/predicted_structures, outputs/alphafold_results, "
                             "outputs/simulated_pdbs")
    parser.add_argument("--style", type=str, default="cartoon",
                        choices=["cartoon", "stick", "sphere", "line"],
                        help="3Dmol.js render style (default: cartoon)")
    parser.add_argument("--spin", action="store_true",
                        help="Auto-rotate the structure in the exported viewer")
    parser.add_argument("--output_html", type=str, default=None,
                        help="Path to write the standalone interactive HTML "
                             "(default: outputs/figures/<job_name>_3d.html)")
    args = parser.parse_args()

    structure_path = resolve_structure_path(args.structure)
    logger.info("Resolved structure: %s", structure_path)

    view, chain_ids = render_structure(structure_path, style=args.style, spin=args.spin)
    confidences = load_summary_confidences(structure_path)

    job_name = os.path.basename(structure_path)
    for suffix in ("_model_0.cif", "_model_0.pdb", ".cif", ".pdb"):
        if job_name.endswith(suffix):
            job_name = job_name[: -len(suffix)]
            break

    output_html = args.output_html or os.path.join("outputs", "figures", f"{job_name}_3d.html")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    html = view.write_html()
    if confidences and "chain_iptm" in confidences:
        caption = (
            f"<p style='font-family:sans-serif'>chains: {chain_ids} &middot; "
            f"chain_iptm: {confidences['chain_iptm']}</p>"
        )
        html = f"{html}\n{caption}"

    with open(output_html, "w") as f:
        f.write(html)

    logger.info("Wrote interactive 3D viewer to %s", output_html)
    print(f"3D structure viewer written to: {output_html}")


if __name__ == "__main__":
    main()
