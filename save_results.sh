#!/bin/bash

${1:?'Define wich network to save'}
    
echo 'Copying files to results directory'

newdir=results/$2/

if [[ $# == 2 ]]
then

   if [ ! -d results/$2 ]
   then
    mkdir results/$2
   fi  

   cp ModelWeights/$1-t?e??.h5 $newdir
   cp ModelWeights/$1-weights.h5 $newdir
   cp TrainedModels/$1-model.h5 $newdir
   cp cache/metadata.pik $newdir
   cp cache/split_ratio.pik $newdir
   cp cache/data_dims.pik $newdir
   cp logs/confusion_matrix-TILS-* $newdir
   cp logs/test_pred.pik $newdir
   mv slurm-* $newdir
else
   echo 'Define the directory name to store data'
fi
