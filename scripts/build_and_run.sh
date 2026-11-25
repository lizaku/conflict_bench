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
# Run from the PARENT of the conflict_bench folder (the code imports itself
# as conflict_bench.*).  Both commands resume if the job is restarted.
MODEL=${MODEL:-microsoft/Phi-3.5-mini-instruct}

# 1. build every instance for this model -> conflict_bench/bench/<model>/
python -m conflict_bench.construct.build --model "$MODEL"

# 2. pilot: the methods on ~100 instances of the four core recipes
python -m conflict_bench.harness --model "$MODEL" --limit 100 --name pilot100
