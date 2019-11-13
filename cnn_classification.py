import data_helpers as dhp
import numpy as np
from os.path import exists
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers import Concatenate
np.random.seed(0)


# vocab 저장, 이후 load 사용
def build_vocab(txt_path, vocab_path, d_type):
    with open(txt_path, 'r', encoding='utf8') as v:
        sentences = v.readlines()
        sentences = dhp.clean_sentences(sentences, d_type)
        dhp.build_vocab(sentences, vocab_path)


# CNN 학습코드
# Based on https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
def train(pos_path, neg_path, vocab_path, d_type, model_path, w2v_path='', is_save=False):
    # Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
    model_type = "CNN-rand"  # CNN-rand|CNN-non-static|CNN-static

    # Model Hyperparameters
    embedding_dim = 50
    filter_sizes = (3, 8)
    num_filters = 10
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50

    # Training parameters
    batch_size = 64
    num_epochs = 10

    # Prepossessing parameters
    sequence_length = 400
    max_words = 5000

    # Data Preparation
    print("Load data...")
    x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv = dhp.prepare_train_set(pos_path,
                                                                                         neg_path,
                                                                                         vocab_path,
                                                                                         d_type)
    if sequence_length != x_test.shape[1]:
        print("Adjusting sequence length for actual size")
        sequence_length = x_test.shape[1]

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    print("Model type is", model_type)
    if model_type in ["CNN-non-static", "CNN-static"]:
        embedding_weights = None
        if exists(w2v_path):
            embedding_weights = dhp.load_w2v_weight(w2v_path, vocabulary_inv)

            if model_type == "CNN-static":
                x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
                x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
                print("x_train static shape:", x_train.shape)
                print("x_test static shape:", x_test.shape)

    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # Build model
    if model_type == "CNN-static":
        input_shape = (sequence_length, embedding_dim)
    else:
        input_shape = (sequence_length,)

    model_input = Input(shape=input_shape)

    # Static model does not have embedding layer
    if model_type == "CNN-static":
        z = model_input
    else:
        z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Initialize weights with word2vec
    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
              validation_data=(x_test, y_test), verbose=2)

    if is_save:
        model.save(model_path)

    return model


def test(model_path, vocab_path, d_type):
    test_sentences = [
        '아 감기에 걸려서 너무 아파... 뭐 먹어야 나을 수 있는거야?',
        '지금 이 겜 안하면 미친놈이다 진짜 개 혜자야',
        'ㅜㅜㅜ BTS 이번 콘서트 독감 걸렸다는데 고생 많았어 ㅠㅠㅠㅠ',
        '이건 진짜 아님',
        '독감 독감 너무 아파 독감 걸렸어',
        '지금 두통이 너무 심해 혹시 약 사다줄 사람 있어? 역 근처까지만 와줘도 너무 고마울 것 같아 ㅠㅠㅠㅠㅠ',
        '야 지금 피방 가자',
        '아 존나 짜증나',
        '헤헤헤 너무 기뻐 나 오늘 수학경시대회에서 상 받음',
        '@hee_PDP 헐 ㅠㅠㅠㅠㅠㅠㅠㅠ 독감 걸렷어 ㅠㅠ?',
        '일 잘 하지도 못하면서...왜 계약 연장 했어? #박지훈건강_피드백해  인기 많은 완아완 잡아두고 싶어서...? 와엠씨는 아티스트 건강도'
        + '무시하고 파트나 무대 배분하는거도 못하고 피드백하라고 하면 눈막귀막 ㄹㅇ기적의 소속사다ㅋㅋㅋ',
        '열어분 건강은 미리 챙깁시다......직음 저는 머리 아파서 아무것도 못하는 사람....규칙적으로 생활하라는ㅇ게 괜ㅊ히 말이 잇는게 아니올시다.....'
    ]

    model = load_model(model_path)
    clean = dhp.clean_sentences(test_sentences, d_type)

    vocabulary, _ = dhp.load_vocab(vocab_path)
    converted = dhp.convert_data(clean, vocabulary)
    prd = model.predict(converted)
    print(prd)


if __name__ == '__main__':
    d_type = 'sns'     # news / sns
    sentences_path_for_vocab = 'file path for building vocabulary'
    pos_path = 'file path for positive'
    neg_path = 'file path for negative'
    vocab_path = 'file path for vocab'
    is_save = True  # 학습 이후 저장여부
    model_path = 'save path for model'  # 저장한다면 저장 경로, 확장자 h5
    w2v_path = 'file path for w2v in gensim'    # gensim의 word2vec을 사용할 경우 경로

    build_vocab(sentences_path_for_vocab, vocab_path, d_type)   # 첫 vocab 생성 후 실행할 필요 없음. 주석처리 할 것.
    train(pos_path, neg_path, vocab_path, d_type, model_path, w2v_path=w2v_path, is_save=is_save)
    test(model_path, vocab_path, d_type)









