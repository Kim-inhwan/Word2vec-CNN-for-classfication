## Word2vec convolutional neural networks for classification of news articles and tweets

### Introduction
 딥 러닝 기술이 다양한 분야에 적용되면서 온라인 뉴스 그리고 트위터와 같은 마이크로 블로그는 딥 러닝 학습을 위한 중요한 자원으로 주목 받고 있다. 그러나, 웹에 업로드 되는 뉴스 기사 혹은 트윗들은 대부분 학습에 불필요한 데이터를 포함한다, e.g., 광고를 포함하거나 무의미한 내용을 담고 있는 뉴스 및 트윗. 본 논문은 온라인 뉴스 기사 혹은 트윗에서 불필요한 데이터를 분류하기 위해 word2vec Convolutional Neural Networks(CNNs)를 사용했고 word2vec의 두 가지 학습 방법에 따른 성능을 비교했다. word2vec의 두 가지 학습 방법인 CBOW와 Skip-gram 중 CBOW의 성능이 더 안정적인 것을 확인할 수 있었으며, CBOW의 경우 뉴스 기사 그리고 Skip-gram의 경우 트윗에서 더 좋은 성능을 보였다.

### Background Knowledge

