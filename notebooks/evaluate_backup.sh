# Run with $ bash scripts/pretrain_P5_base_beauty.sh 4

#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0,2
#
#name=beauty-base
#
#output=snap/$name
dataset=$1
model=$2
cuda_list=$3
port=$4
warmup_ratio=$5
learning_rate=$6
seed=$7
cuda_num=${#cuda_list[@]}
output="notebooks/${dataset}/${model}"

log_name="${dataset}_${model}_sequential"
llm_root_dir="/data/home/lzq/lvzheqi/ICLR2023/LLM/llm"
#llm_root_dir="../../../llm"
backbone="${llm_root_dir}/${model}"
export CUDA_VISIBLE_DEVICES=${cuda_list}

#PYTHONPATH=$PYTHONPATH:./src \
python notebooks/main_test_beauty_small.py \
        --seed ${seed} \
        --train ${dataset} \
        --valid ${dataset} \
        --test ${dataset} \
        --batch_size 32 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'sequential' \
        --backbone ${backbone} \
        --output ${output} > ${log_name}.log
#        --epoch 10 \
#        --max_text_length 512 \
#        --gen_max_length 64 \
#        --whole_word_embed > ${log_name}.log
