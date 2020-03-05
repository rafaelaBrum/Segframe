#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="EN-5"
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:/pylon5/ac3uump/alsm/lib64/python3.6/site-packages:$PYTHONPATH

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /pylon5/ac3uump/alsm/venv/bin/activate

module load cuda/9.0
module load openslide/3.4.1

echo '[START] training'
date +"%D %T"

time python3 Utils/WSITile.py -ds ../data/wsis/ -od ../data/extracted -hm ../data/til_maps/TIL_maps_after_thres_v1/ -mp 3 -ps 100 -wr 0.25 -txt_label -hmc

echo '[FINAL] done training'

deactivate 

date +"%D %T"


