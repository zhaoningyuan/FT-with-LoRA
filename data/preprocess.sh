#!/bin/bash

for dataset in train dev test_public
do
    source_file="pCLUE_${dataset}.json"
    target_file="pCLUE_${dataset}.csv"
    python prepare_dataset.py --source_file $source_file --target_file $target_file
done