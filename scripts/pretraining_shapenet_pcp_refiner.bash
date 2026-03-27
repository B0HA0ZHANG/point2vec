#!/bin/bash
#SBATCH --job-name=p2v-pre-sh-pcpr
#SBATCH --no-requeue
#SBATCH --time=2-00:00
#SBATCH --begin=now
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

python -m point2vec.pretrain_pcp_refiner fit --config "configs/pretraining/shapenet_pcp_refiner.yaml" --config "configs/wandb/pretraining_shapenet_pcp_refiner.yaml" "$@"
