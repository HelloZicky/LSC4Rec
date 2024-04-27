# Collaborative LLMs and Small Recommender Models for Device-Cloud Recommendation

This is the implementation of the "Collaborative LLMs and Small Recommender Models for Device-Cloud Recommendation"


---
### LLM&SRM Independent Training
> LLM

bash train.sh 

> SRM

bash train_small.sh 小模型t-2训练，
---
### LLM&SRM Collaborative Training
bash train_collaboration.sh 融合t-2训练,

---
### SRM Real-time Retraining
bash train_small_retrain.sh

---
### LLM&SRM Collaborative Inference&Request
> set ratio

bash colla_test.sh

### Random Request
> set ratio

python notebooks/test_filter_collaboration_dcc_t2.py
