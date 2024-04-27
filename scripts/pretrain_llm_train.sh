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
output="checkpoint_large/${dataset}/${model}_${type}"

log_name="checkpoint_large/${dataset}_${model}_${type}"
#llm_root_dir="/data/home/lzq/lvzheqi/ICLR2023/LLM/llm"
llm_root_dir="/data/zhantianyu/LLM/llm"
#llm_root_dir="../../../llm"
backbone="${llm_root_dir}/${model}"
export CUDA_VISIBLE_DEVICES=${cuda_list}

#PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=${cuda_num} \
    --master_port ${port} \
    src/pretrain_recommendation.py \
        --distributed --multiGPU \
        --dcc True \
        --seed ${seed} \
        --train ${dataset} \
        --valid ${dataset} \
        --batch_size 32 \
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
        --framework large_language_model \
        --type ${type} \
        --whole_word_embed > ${log_name}.log
