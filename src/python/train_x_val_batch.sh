#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument 1 supplied: either train or eval"
fi

if [ -z "$2" ]
then
    echo "No argument 2 supplied: choose dataset folder, e.g. mturk_partial, lab, mturk_full"
    exit 1
fi

features=( "bow previous_action nlu"
    "wrd_emb_gn bow previous_action nlu"
    "bow action_mask previous_action nlu"
    "wrd_emb_gn bow action_mask previous_action nlu")


if [[ $2 == *"mturk"* ]]
then
    train_dir="mturk"
    # orca_dact.lm is n-gram model with both datasets, 63 dialogues from each
    dam="/scratch/orca/orca_dact.lm"
else
    train_dir="in-lab"
    dam="/scratch/orca/orca_dact.lm"
fi

if [ "$1" = "train" ]
then
    echo Training...

    for feature in "${features[@]}"
    do
        echo ${feature}
        python train_x_validation.py -d ../../data/${train_dir}/ --experimental-condition $2 --features ${feature}
        # check if succeed
        if [ $? -eq 0 ]
        then
            echo OK: $2/${feature}
        else
            echo ERROR
            echo Command: train_x_validation.py -d ../../data/${train_dir}/ --experimental-condition $2 --features ${feature}
            exit 1
        fi
    done;
fi

if [ "$1" = "eval" ]
then
    echo Evaluating with LM...

    for feature in "${features[@]}"
    do
        train_x_validation.py -d ../../data/${train_dir}/ --experimental-condition $2 --features ${feature} -dam ${dam} -g
        # check if succeed
        if [ $? -eq 0 ]
        then
            echo OK: $2/${feature}
            read -p "Press return to continue"
        else
            echo ERROR
            echo Command: train_x_validation.py -d ../../data/${train_dir}/ --experimental-condition $2 --features ${feature} -dam ${dam} -g
            exit 1
        fi
    done;
fi