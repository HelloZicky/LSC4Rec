# Collaborative LLMs and Small Recommender Models for Device-Cloud Recommendation

This is the implementation of the "Collaborative LLMs and Small Recommender Models for Device-Cloud Recommendation"


---
### LLM&SRM Independent Training
> LLM

bash train.sh 

> SRM

bash train_small.sh
---
### LLM&SRM Collaborative Training
bash train_collaboration.sh 

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
