#!/bin/bash
#
#SBATCH --job-name=conflict_bench
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2) 

# --- resources ------------------------------------------------------
#SBATCH --gres=gpu:1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-20:00:00             # walltime D-HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load CUDA
ml releases/2023a; module load JupyterLab Transformers; source ../../shaprag/bin/activate

# --- run ------------------------------------------------------------
python run_pilot.py