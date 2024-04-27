#!/bin/bash
date
#sleep 14400
#sleep 300
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("beauty" "cds" "electronic")
#dataset_list=("beauty" "cds" "electronic" "yelp")
#dataset_list=("beauty" "sports" "toys" "yelp")
#dataset_list=("beauty" "sports" "yelp")
#dataset_list=("sports" "yelp")
dataset_list=("beauty" "toys")
#dataset_list=("toys")
#dataset_list=("yelp")
#dataset_list=("sports")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(0 1 2)
cuda_num_list=(0 1)
#cuda_num_list=(1)
#cuda_num_list=(2 3 4)
#cuda_num_list=(4 5 7)
#cuda_num_list=(1)
#cuda_num_list=(2)
#cuda_num_list=(3)
echo ${line_num_list}
seed=2022
length=${#dataset_list[@]}
#for dataset line_num in zip()
for ((i=0; i<${length}; i++));
#for i in {0..${length}-1};
do
{
    dataset=${dataset_list[i]}
#    line_num=${line_num_list[i]}
    cuda_num=${cuda_num_list[i]}
#    for model in din gru4rec sasrec
#    for model in t5-base t5-small
    for model in t5-small
    do
    {
#        for split in train evaluate
#        for split in train
#        do
#          {
#              for type in all traditional sequential rating
#              for type in all traditional
#              for type in all
#              for type in traditional
#              for type in sequential
#              do
#                {
#                  bash small_test.sh ${dataset} ${model}
#
##                  cd ../
#                } &
#              done
#              for framework in small_recommendation_model
#              for framework in large_language_model
              for framework in collaboration
              do
                {
#                  for losses in sequential traditional all
#                  for losses in sequential traditional
#                  for losses in traditional
                  for losses in sequential
                  do
                    {
#                        for small_model in din gru4rec sasrec
#                        for small_model in sasrec
#                        for small_model in gru4rec
                        for small_model in din
                        do
                          {
#                              for type_small in base duet
                              for type_small in base
                              do
                                {
#                                    for rank_model in small both
                                    for rank_model in small
                                    do
                                      {
        #                                  bash func_collaboration_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${losses} ${small_model} ${type_small}
#                                          bash func_collaboration_filter_dcc_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${losses} ${small_model} ${type_small} ${rank_model}
                                          bash func_collaboration_filter_dcc_t3_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${losses} ${small_model} ${type_small} ${rank_model}
                                      } &
                                    done
                                } &
                              done
                          } &
                        done
                    } &
                  done

#                  cd ../
                } &
              done
#          } &
#        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
