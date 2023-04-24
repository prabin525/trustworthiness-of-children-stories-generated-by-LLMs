#!/bin/bash

#SBATCH --job-name=generate_stories

##HOPPER
##SBATCH --qos=gpu --partition=gpuq --gres=gpu:A100.80gb:1
#SBATCH --qos=gpu --partition=gpuq --gres=gpu:3g.40gb:1

#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err

#SBATCH --mem=200G
#SBATCH --time=5-00:00:00

#SBATCH --nodes=1 --ntasks-per-node=1

cd /scratch/pbhanda2/projects/children_stories_gen
source myenv/bin/activate

export TRANSFORMERS_CACHE=/scratch/pbhanda2/huggingface/
export TORCH_MODEL_ZOO=/scratch/pbhanda2/.torch/torch_model_zoo/
export TORCH_HOME=/scratch/pbhanda2/.torch/torch_home/
export STANZA_RESOURCES_DIR=/scratch/pbhanda2/.stanza/

python generate_stories.py --model $1
