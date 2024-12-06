# plink2hdf5

## Overview
**plink2hdf5** is a computational workflow designed to process and transform genetic data from PLINK file formats (PGEN/BED) into HDF5 format, split the data into training and testing sets, and adjust the data for covariates. This workflow is implemented using **Snakemake**, providing a robust, modular, and reproducible pipeline. It is mainly suited to taking QC'd genotype data and converting to hdf5 files for downstream machine learning. Designed for SLURM-managed HPC clusters but can be adapted for others.

## Core Workflow

- **PLINK File Conversion**: Converts PLINK genotype data (`.pgen`, `.bed`, etc.) into a `.raw` format using `plink2` for further processing.
- **Data Shuffling**: Shuffles raw genotype data to randomize its order for unbiased downstream analysis using `shuf`.
- **HDF5 Conversion**: Transforms shuffled genotype data into HDF5 format using efficient chunking and `dask` for scalability.
- **Train/Test Splitting**: Splits the HDF5 file into training and testing subsets based on specified proportions and random seeds.
- **PLINK File Splitting**: Creates training and testing subsets of the original PLINK files, maintaining compatibility with PLINK-based tools.
- **Covariate Adjustment**: Adjusts genotype data and outcome (as per Zhou et al.) in hdf5 files for covariates, enabling statistical corrections before downstream analyses.
- **Intermediate Cleanup**: Efficient management of intermediate files, ensuring minimal storage usage during workflow execution.
- **Modularity**: Configurable input/output paths, file formats, and computation parameters via a YAML configuration file.

## Ongoing work

Still needs:

- report of train/test splits via jupyter notebook
- python tests
- workflow tests

# Installation

## Download and install the requirements
```
git clone https://github.com/seafloor/plink2hdf5.git
cd plink2hdf5

# install a base snakemake env for runninng the workflow
module load your/base/conda/module
conda config --set channel_priority strict
conda create -f requirements.yaml
conda activate snakemake
```

## Install the workflow env on the head node

Install the separate conda env that will be loaded on compute nodes to run the pipelines. Must be done now on the head node, as most clusters have no internet access on compute nodes:

```
cd workflow
conda create -f envs/python_env.yaml
```

## Customise the config file
Everything is managed via a config file - you shouldn't need to touch the snakefile. Copy the example and edit:
```
cp example_config.yaml config.yaml
vim config.yaml # add your own paths etc.
```

# Running

## Check the workflow (dry run)
Check a dry run to see the workflow:
```
snakemake -np --use-envmodules --cores 1
```

## Run the full workflow on slurm
```
# make sure config.yaml exists and is correct first!
snakemake -p --use-envmodules --cores 1 --executor slurm --workflow-profile profiles/slurm --jobs 1
```

## Debugging possible conda fixes
Some systems have issues with the conda version not being found by snakemake. You make need to run the lines below to fix this.
```
~/.conda/envs/snakemake/bin/conda init
source ~/.bashrc
```

---

## License

This project is licensed under the **MIT License**. See the license file for details.

## Contributions

Contributions are welcome! If you have ideas, improvements, or bug fixes, please submit a pull request or open an issue. Thank you for helping make this project better!