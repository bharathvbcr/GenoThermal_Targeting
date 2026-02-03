from setuptools import setup, find_packages

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
)
