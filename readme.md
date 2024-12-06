# plink2hdf5

# installation

## download and install the requirements
pull repo
cd repo

module load your/base/conda/module
conda config --set channel_priority strict
conda create -f requirements.yaml
conda activate snakemake
cd workflow

## possible conda fixes
~/.conda/envs/snakemake/bin/conda init
source ~/.bashrc

## make sure any python envs are installed already from the head node
### if installing via snakemake
snakemake --use-conda --conda-create-envs-only --conda-frontend conda --cores 1

### if installing directly
conda create -f envs/python_env.yaml

# Running
cd workflow

## check the workflow (dry run)
snakemake -np --use-envmodules --cores 1

## run the full workflow on slurm
snakemake -p --use-envmodules --cores 1 --executor slurm --workflow-profile profiles/slurm --jobs 1


## run it locally in an interactive job
salloc --job-name=snakemake --time=01:00:00 --nodes=1 --ntasks=1 --tasks-per-node=1 --mem-per-cpu=20G
srun --pty /bin/bash
module load your/base/conda/module
conda activate snakemake

snakemake -p --use-envmodules --cores 1