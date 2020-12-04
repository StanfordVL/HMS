#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=napoli-gpu
#################
#number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#################
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/cvgl2/u/andreyk/projects/semantic_mech_search/semantic-xray/logs/slurm/%j.out
#################
# a file for errors from the job
#SBATCH --error=/cvgl2/u/andreyk/projects/semantic_mech_search/semantic-xray/logs/slurm/%j.err
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 4.
#SBATCH -c 16
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=andreyk@stanford.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal batch commands
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/andreyk/.mujoco/mjpro150/bin
source /cvgl2/u/andreyk/projects/semantic_mech_search/env/bin/activate
cd /cvgl2/u/andreyk/projects/semantic_mech_search/semantic-xray
echo $PWD
echo "Starting training ours_no_images" 
python scripts/train.py --model_type ours_no_images
# done
echo "Done"
exit 0
