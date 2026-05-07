from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="geno_thermal_targeting",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["alphagenome_utils", "alphafold_utils"],
    install_requires=[
        "requests",
        "pandas",
        "matplotlib",
        "ipywidgets",
        "numpy",
        "jupyter",
        "seaborn",
        "py3Dmol",
        "biopython",
    ],
    scripts=[
        "genomic_discovery.py",
        "ligand_designer.py"
    ],
    description="Toolbox for Geno-Thermal Targeting nanoparticle design",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
)
