#!/bin/bash

dataset=$1
framework=$2
small_recommendation_model=$3
cuda_list=$4
losses=$5
#epoch=$6

export CUDA_VISIBLE_DEVICES=${cuda_list}
cd ../
#python notebooks/test_beauty_small_sequential.py ${dataset} ${framework} ${small_recommendation_model} ${losses} ${epoch}
python notebooks/test_beauty_small_sequential.py ${dataset} ${framework} ${small_recommendation_model} ${losses}