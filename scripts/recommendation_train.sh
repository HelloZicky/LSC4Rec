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
recommendation_model=$8
framework=$9
type_small=${10}
#losses_small=${11}
type=${11}
#full_dataset=${11}
#output="checkpoint/${dataset}/${model}_all"
#output="checkpoint/${dataset}/${model}_${recommendation_model}_${framework}"
output="checkpoint_small_dcc/${dataset}/${type_small}_${recommendation_model}_${type}"

#log_name="checkpoint/${dataset}_${model}_all"
log_name="checkpoint_small_dcc/${dataset}/${type_small}_${recommendation_model}_${type}"
llm_root_dir="/data/zhantianyu/LLM/llm" # 16
#llm_root_dir="/data/lvzheqi/lvzheqi/ICLR2023/LLM/llm" # 110
#llm_root_dir="../../../llm"
backbone="${llm_root_dir}/${model}"
export CUDA_VISIBLE_DEVICES=${cuda_list}
#cd scripts/
#ARCH_CONF_FILE="scripts/configs/amazon_${dataset}_conf.json"
#ARCH_CONF_FILE="scripts/configs/${full_dataset}_conf.json"
ARCH_CONF_FILE="scripts/configs/${dataset}_conf.json"
#PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=${cuda_num} \
    --master_port ${port} \
    src/pretrain_recommendation.py \
        --distributed --multiGPU \
        --dcc False \
        --seed ${seed} \
        --train ${dataset} \
        --valid ${dataset} \
        --batch_size 128 \
        --optim adamw \
        --warmup_ratio ${warmup_ratio} \
        --lr ${learning_rate} \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses ${type} \
        --type ${type} \
        --backbone ${backbone} \
        --output ${output} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --small_recommendation_model ${recommendation_model} \
        --framework ${framework} \
        --arch_config=${ARCH_CONF_FILE} \
        --whole_word_embed > ${log_name}.log
