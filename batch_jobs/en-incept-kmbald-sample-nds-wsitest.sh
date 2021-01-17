#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 35:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="EN-143"
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:/pylon5/ac3uump/alsm/lib64/python3.6/site-packages:$PYTHONPATH

if [ ! -d $LOCAL/test ]
then
    mkdir $LOCAL/test;
fi

cd $LOCAL/test/

#echo 'Uncompressing data to LOCAL'

#cp -r /pylon5/ac3uump/alsm/active-learning/data/extracted $LOCAL/test/
#tar -xf lym_cnn_training_data.tar -C $LOCAL/test

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /pylon5/ac3uump/alsm/venv/bin/activate

module load cuda/9.0

echo '[START] training'
date +"%D %T"

time python3 main.py -i -v --al -strategy EnsembleTrainer -predst /pylon5/ac3uump/alsm/active-learning/data/nds300 -split 0.90 0.01 0.09 -net EFInception -data CellRep -init_train 500 -ac_steps 20 -emodels 3 -ac_function kmng_uncert -un_function ensemble_bald -acquire 200 -d -e 50 -b 60 -tdim 240 240 -clusters 20 -out logs/ -cpu 6 -gpu 2 -tn -sv -nsw -wpath results/$DIRID -model_dir results/$DIRID -logdir results/$DIRID -cache results/$DIRID -pca 50 -sample 2000 -wsi_split 5 -pred_size 15000 -load_train -spool 2 -k -tnet Inception -tnpred 2 -phi 2 -f1 30 -lr 0.0001 -wsilist TCGA-BL-A13J-01Z-00 TCGA-FR-A728-01Z-00 TCGA-EE-A2MH-01Z-00 TCGA-C5-A1MH-01Z-00 TCGA-US-A77G-01Z-00 -wsimax 1.0 1.0 1.0 1.0 0.8

echo '[FINAL] done training'

deactivate 

date +"%D %T"


