# 🚀 Collaborative LLMs and Small Recommender Models for Device-Cloud Recommendation (KDD 2025)

[![Static Badge](https://img.shields.io/badge/arXiv-2501.05647-logo?logo=arxiv&labelColor=red&color=peachpuff)](https://arxiv.org/abs/2501.05647) [![Static Badge](https://img.shields.io/badge/Scholar-LSC4Rec-logo?logo=Googlescholar&color=blue)](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=Collaboration+of+Large+Language+Models+and+Small+Recommendation+Models+for+Device-Cloud+Recommendation&btnG=) [![Static Badge](https://img.shields.io/badge/Semantic-LSC4Rec-logo?logo=semanticscholar&labelcolor=purple&color=purple)](https://www.semanticscholar.org/paper/Collaboration-of-Large-Language-Models-and-Small-Lv-Zhan/6d8647203161ae4b700138f4ef1f6d4e39648c4c) [![Static Badge](https://img.shields.io/badge/GitHub-LSC4Rec-logo?logo=github&labelColor=black&color=lightgray)](https://github.com/HelloZicky/LSC4Rec) ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6d8647203161ae4b700138f4ef1f6d4e39648c4c%3Ffields%3DcitationCount&style=social&logo=semanticscholar&labelColor=blue&color=skyblue&cacheSeconds=360)

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
