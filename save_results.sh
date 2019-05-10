#!/bin/bash

${1:?'Define wich network to save'}
    
echo 'Copying files to results directory'

newdir=''

if [ $# == 3 ]
then
    mkdir results/$2
    newdir=results/$2
else
    
    itn = 0
    for it in results/; do
	if [ -d results/$it ]
	then
	    itn = ${curr#-AT}
	fi

	if [ itn < curr ]
	then
	    itn = curr
	fi
    done
    
    itn += 1
    mkdir results/$itn
    newdir=results/$itn
fi

cp ModelWeights/$1-t?e??.h5 $newdir
cp ModelWeights/$1-weights.h5 $newdir
cp TrainedModels/$1-model.h5 $newdir
cp cache/metadata.pik $newdir
cp cache/split_ratio.h5 $newdir
cp logs/confusion_matrix-TILS-* $newdir
cp logs/test_pred.pik $newdir
