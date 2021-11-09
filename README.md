# aiffel_NLP
텍스트 간 유사도 측정 및 자연어 학습 Task에 대하여 공부합니다.

- [[D-1]Text_Preprocessing](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-1%5DText_preprocess.ipynb) :
  - regular expression으로 전처리하고, `Mecab`, `Khaii`, `Wordpiece model` 로 Tokeninzing을 진행합니다.
- [[D-2]Tokenizing](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-2%5DTokenizing(SentencePiece%2C%20BPE%2C%20Mecab).ipynb) : 
  - space/형태소 기반 Tokenizing에 대하여 이해하고, Google의 `SentencePiece` 모델, 별도 함수 구현등의 방법으로 Tokeninzing을 진행하는 프로젝트입니다.
- [[D-3]similarity_TF_IDF](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-3%5Dsimilarity_TF_IDF.ipynb) :
  - 텍스트 분포를 이용한 텍스트의 벡터화 방법(DTM, TF-IDF)을 실습을 통해 익혀봅니니다.
- [[D-4]News_classification](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-4%5DNews_classification.ipynb) :   
   Confusion Matrix에 대하여 이해하고, 뉴스 카테고리 다중 분류 실습을 통해 성능을 비교해 봅시다.
  - `DTM`, `TF-IDF`, `SVM`등의 Linear한 모델
  - `Naive Bayes Classifier`, `CNB`, `SoftMax` 등의 확률론 또는 딥러닝 알고리즘
  - `DecisionTree`, `RandomForest`, `GradientBoost` 등의 앙상블 모델과 Voting을 통한 개선
- [[D-5]Text_One_hot_encoding, Embedding](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-5%5DText_One_hot_encoding%2C%20Embedding.ipynb):  
  - Embedding에 대하여 이해하고, `Word2Vec` 모델로 벡터화를 진행합니다.
  - CBow와 Skip-gram에 대하여 이해합니다.
  - FastText와 Glove에 대하여 공부합니다.
- [[D-8]huggingface_basic_manual](https://github.com/crosstar1228/NLP_and_Recommendation/blob/main/%5BD-8%5Dhuggingface_basic_manual.ipynb): NLP 학습 Framework인 huggingface에 대하여 이해하고, pretrained model을 불러오는 과정에 대하여 알아봅니다.