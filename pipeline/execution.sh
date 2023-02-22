#!/bin/bash
#SBATCH --job-name=dia-ret-data
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=100GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/mnadar2s/perl5/MohanRajRnD/pipeline/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/mnadar2s/perl5/MohanRajRnD/pipeline/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/rnd

# locate to your root directory
cd /home/mnadar2s/perl5/MohanRajRnD/pipeline

# run the script
python main.py

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml