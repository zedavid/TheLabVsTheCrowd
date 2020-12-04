#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument 1 supplied: choose dataset folder, e.g. mturk_partial, lab, mturk_full"
    exit 1
fi

if [ -z "$2" ]
then
    echo "No argument 2 supplied:please provide a word2vec/glove embedding file"
    exit 1
fi

features=( "bow previous_action nlu"
    "wrd_emb bow previous_action nlu"
    "bow action_mask previous_action nlu"
    "wrd_emb bow action_mask previous_action nlu")

if [[ $1 == *"mturk"* ]]
then
    data_dir="mturk"
else
    data_dir="in-lab"
fi

if [ "$#" -eq 3 ]; then
  if [[ $3 == *"mturk"* ]]
  then
      test_data_dir="mturk"
  else
      test_data_dir="in-lab"
  fi
fi

echo Extracting features...

for feature in "${features[@]}"
do
    if [ "$#" -eq 3 ]; then
      echo "test data"
      python extract_features.py -tr ../../data/${data_dir}/ --test_dir ../../data/${test_data_dir}/ --experimental-condition $1 --features ${feature} -ef $2
    else
      python extract_features.py -tr ../../data/${data_dir}/ --experimental-condition $1 --features ${feature} -ef $2
    fi
    # check if succeed
    if [ $? -eq 0 ]
    then
        echo OK: $2/${feature}
    else
        echo ERROR
        echo Command: extract_features.py -tr ../../data/${data_dir}/ --experimental-condition $1 --features ${feature} --embeddings_file $2
        exit 1
    fi
done;
