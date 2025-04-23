#!/bin/bash

#SBATCH --job-name=BTS-Att
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=2
#SBATCH --mem=420G
#SBATCH --time=34:00:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=an517162@ucf.edu

echo "Job started at $(date)" >> slurm_logs/slurm_${SLURM_JOB_ID}.out
cd /lustre/fs1/home/cap5516.student5/CAP5516-MIC/BraTS/
echo "Current directory: $(pwd)" >> slurm_logs/slurm_${SLURM_JOB_ID}.out

if [ -d "/lustre/fs1/home/cap5516.student5/CAP5516-MIC/BraTS/" ]; then
    echo "Directory exists and is accessible" >> slurm_logs/slurm_${SLURM_JOB_ID}.out
else
    echo "Directory not found or inaccessible" >> slurm_logs/slurm_${SLURM_JOB_ID}.err
fi

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-11.8.0
module load gcc/gcc-12.2.0
module list >> slurm_logs/slurm_${SLURM_JOB_ID}.out 2>> slurm_logs/slurm_${SLURM_JOB_ID}.err

source brain_seg_env/bin/activate
echo "Virtual environment activated" >> slurm_logs/slurm_${SLURM_JOB_ID}.out

export WANDB_API_KEY="9524dd5ae08e91b83c094966b84fc1db7c6ce03c"
export OPENBLAS_NUM_THREADS=64

mkdir -p slurm_logs

JOB_ID=$SLURM_JOB_ID

echo "Python version:" >> slurm_logs/slurm_${JOB_ID}.out
python --version >> slurm_logs/slurm_${JOB_ID}.out 2>> slurm_logs/slurm_${JOB_ID}.err
echo "Starting main.py..." >> slurm_logs/slurm_${JOB_ID}.out
python main.py --config config/hyperparams.json --model AttUNet >> slurm_logs/slurm_${JOB_ID}.out 2>> slurm_logs/slurm_${JOB_ID}.err
#choices=["StdUNet", "AttUNet", "both"]
echo "main.py completed." >> slurm_logs/slurm_${JOB_ID}.out

if [ ! -z "$(grep 'using W&B mode: offline' slurm_logs/slurm_${JOB_ID}.out)" ]; then
    echo "W&B ran in offline mode. Sync logs later with: 'wandb sync /path/to/outputs_<model>/wandb/offline-run-*'" >> slurm_logs/slurm_${JOB_ID}.out
fi

deactivate
echo "Job completed at $(date)" >> slurm_logs/slurm_${JOB_ID}.out