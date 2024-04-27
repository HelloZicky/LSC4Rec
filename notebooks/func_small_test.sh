#!/bin/bash

dataset=$1
framework=$2
small_recommendation_model=$3
cuda_list=$4
type=$5
export CUDA_VISIBLE_DEVICES=${cuda_list}
cd ../
python notebooks/test_beauty_small_sequential_small.py ${dataset} ${framework} ${small_recommendation_model} ${type} ${type} ${epoch}