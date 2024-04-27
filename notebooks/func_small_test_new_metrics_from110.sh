#!/bin/bash

dataset=$1
framework=$2
small_recommendation_model=$3
cuda_list=$4
type=$5
losses=$6
#epoch=$6
export CUDA_VISIBLE_DEVICES=${cuda_list}
cd ../
python notebooks/test_beauty_small_sequential_small_new_metrics_from110.py ${dataset} ${framework} ${small_recommendation_model} ${type} ${losses}