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
type=$8
cuda_num=${#cuda_list[@]}
small_recommendation_model=$9
type_small=${10}
collaboration_method=${11}
#load="checkpoint/${dataset}/${model}_${type}"
load="checkpoint_large/${dataset}/${model}_${type}"
#load_small="checkpoint/${dataset}/${model}_${type}"
load_small="checkpoint_small_dcc/${dataset}/${type_small}_${small_recommendation_model}_${type}"
#output="checkpoint_collaboration/${dataset}/${model}_${type}_${small_recommendation_model}_${type_small}"
#output="checkpoint_collaboration_${collaboration_method}/${dataset}/${model}_${type}_${small_recommendation_model}_${type_small}"
output="checkpoint_collaboration_aug/${dataset}/${model}_${type}_${small_recommendation_model}_${type_small}"

#log_name="checkpoint_collaboration/${dataset}_${model}_${type}_${small_recommendation_model}_${type_small}"
#log_name="checkpoint_collaboration_${collaboration_method}/${dataset}_${model}_${type}_${small_recommendation_model}_${type_small}"
log_name="checkpoint_collaboration_aug/${dataset}_${model}_${type}_${small_recommendation_model}_${type_small}"
#llm_root_dir="/data/home/lzq/lvzheqi/ICLR2023/LLM/llm"
llm_root_dir="/data/zhantianyu/LLM/llm"
#llm_root_dir="/storage/syma/lvzheqi/llm"
#llm_root_dir="../../../llm"
backbone="${llm_root_dir}/${model}"
ARCH_CONF_FILE="scripts/configs/${dataset}_conf.json"
export CUDA_VISIBLE_DEVICES=${cuda_list}

#PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=${cuda_num} \
    --master_port ${port} \
    src/pretrain_recommendation_collaboration_aug.py \
        --distributed --multiGPU \
        --dcc True \
        --seed ${seed} \
        --train ${dataset} \
        --valid ${dataset} \
        --batch_size 32 \
        --load ${load} \
        --load_small ${load_small} \
        --optim adamw \
        --warmup_ratio ${warmup_ratio} \
        --lr ${learning_rate} \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses ${type} \
        --backbone ${backbone} \
        --output ${output} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --framework collaboration \
        --retrain_type "small" \
        --type ${type} \
        --small_recommendation_model ${small_recommendation_model} \
        --type_small ${type_small} \
        --arch_config=${ARCH_CONF_FILE} \
        --collaboration_method=${collaboration_method} \
        --whole_word_embed > ${log_name}.log
#        --whole_word_embed > ${log_name}.log
#        --fusion_net fusion_net \
