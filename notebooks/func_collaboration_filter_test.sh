#!/bin/bash

dataset=$1
framework=$2
large_language_model=$3
cuda_list=$4
losses=$5
small_recommendation_model=$6
type_small=$7
rank_model=$8
export CUDA_VISIBLE_DEVICES=${cuda_list}
cd ../
python notebooks/test_filter_collaboration.py ${dataset} ${framework} ${large_language_model} ${losses} ${small_recommendation_model} ${type_small} ${rank_model}