## Word2vec convolutional neural networks for classification of news articles and tweets

### Introduction
 딥 러닝 기술이 다양한 분야에 적용되면서 온라인 뉴스 그리고 트위터와 같은 마이크로 블로그는 딥 러닝 학습을 위한 중요한 자원으로 주목 받고 있다. 그러나, 웹에 업로드 되는 뉴스 기사 혹은 트윗들은 대부분 학습에 불필요한 데이터를 포함한다, e.g., 광고를 포함하거나 무의미한 내용을 담고 있는 뉴스 및 트윗. 본 논문은 온라인 뉴스 기사 혹은 트윗에서 불필요한 데이터를 분류하기 위해 word2vec Convolutional Neural Networks(CNNs)를 사용했고 word2vec의 두 가지 학습 방법에 따른 성능을 비교했다. word2vec의 두 가지 학습 방법인 CBOW와 Skip-gram 중 CBOW의 성능이 더 안정적인 것을 확인할 수 있었으며, CBOW의 경우 뉴스 기사 그리고 Skip-gram의 경우 트윗에서 더 좋은 성능을 보였다.

### Background Knowledge
![Model architecture of (A) CBOW and (B) Skip-gram](./img/word2vec_structure.png)
위 그림은 word2vec의 두 학습 방법인 CBOW와 Skip-gram의 모델 구조를 보여준다. Word2vec은 문서 내 단어들 간의 의미론적 유사성을 반영하여 각 단어에 벡터 값을 부여하는 기술이다. CBOW는 인접한 단어들이 등장했을 때 목표 단어가 등장할 확률을 기반으로 학습하고, Skip-gram은 목표 단어가 등장했을 때 인접 단어가 등장할 확률을 기반으로 학습한다.

![The general model of the CNN](./img/cnn_structure.png)
위 그림은 CNNs의 일반적인 모델 구조를 보여준다. Input은 n x n 크기의 커널을 stride 크기에 맞춰 이동시키면서 특징을 추출해 features maps로 값을 전달한다. Features maps에서도 동일하게 커널을 이동시키면서 pooling과정을 진행하여 2차원 혹은 3차원 데이터를 1차원 데이터로 변환한다. 이러한 과정을 통해 2차원 혹은 3차원 데이터는 공간적 데이터 손실 없이 학습 데이터로서 사용될 수 있다.

### Classification model
![CNN architecture with word2vec](./img/model_structure.png)
위 그림은 본 논문에서 사용한 word2vec CNN 모델 구조를 보여준다. 예문으로 사용한 'Ebola virus was found in Congo'는 각 단어 별로 쪼개지고 word2vec에 의해 각 단어에 부여된 벡터 값이 convolutional layer로 입력된다. 이후, max-pooling 과정을 거쳐 fully-connected layer로 입력되고 모델은 해당 문장이 유의미한 문장(Positive)인지 불필요한 문장(Negative)인지 분류한다.

### Experiments
본 논문은 word2vec의 두 학습 방법인 CBOW, Skip-gram 그리고 word2vec을 사용하지 않은 random vector를 사용했을 때 분류 성능을 평가했다. 성능 평가 방법은 Accuracy와 F1 score를 사용했다. 
