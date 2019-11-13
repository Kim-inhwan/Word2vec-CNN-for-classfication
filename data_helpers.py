from os.path import exists
from konlpy.tag import Okt
from multiprocessing import Manager
from collections import Counter
from tqdm import tqdm
import pickle
import itertools
import re
import parmap
import numpy as np
import warnings


"""
문장의 길이를 맞추기 위해 padding함
최대 길이는 트윗의 길이인 140자로 지정
"""


def pad_sentences(sentences, d_type, padding_word="<PAD/>"):
    sequence_length = 140 if d_type == 'sns' else 300

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


"""
트윗, 뉴스의 불필요한 부분을 제거함
해시태그, 리트윗, 이모지 등...
품사별로 분리해서 배열 형태로 리턴
"""


def remove_tag(string, result):
    warnings.filterwarnings(action='ignore')
    # tokenizer = Mecab()
    tokenizer = Okt()  # windows의 경우 Mecab 사용 불가

    string = re.sub(r"@[A-Za-z0-9ㄱ-ㅎ가-힣!@$%^&()_]*", " ", string)
    string = re.sub(r"#[A-Za-z0-9ㄱ-ㅎ가-힣!@$%^&()_]*", " ", string)
    string = re.sub(r"[^A-Za-z0-9ㄱ-ㅎ가-힣!?]", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = tokenizer.pos(string)
    # string = [word for word, pos in string if not pos == 'UNKNOWN']   # for Mecab
    string = [word for word, pos in string if not (pos == 'Unknown' or pos == 'KoreanParticle')]    # for Okt

    result.append(string)


def clean_sentences(sentences, d_type, is_multi=True):
    if is_multi:
        # 멀티 프로세싱을 위한 코드
        sentence_list = Manager().list([])
        parmap.map(remove_tag, sentences, sentence_list, pm_pbar=True, pm_processes=4)
    else:
        sentence_list = []
        for sentence in tqdm(sentences):
            remove_tag(sentence, sentence_list)

    sentence_list = pad_sentences(sentence_list, d_type)

    return sentence_list


"""
vocabulary, vocabulary_inv를 생성한다.
형식은 {"단어":숫자, ... , "단어": 숫자}
inv의 경우 반대
"""


def build_vocab(sentences, save_path=''):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    # process unknown words
    vocabulary['<UNKNOWN/>'] = len(vocabulary)
    vocabulary_inv.append('<UNKNOWN/>')

    if not save_path == '':
        with open(save_path, 'wb') as v:
            pickle.dump({'vocab': vocabulary, 'vocab_inv': vocabulary_inv}, v)

    return vocabulary, vocabulary_inv


def load_vocab(file_path):
    if exists(file_path):
        with open(file_path, 'rb') as v:
            voc = pickle.load(v)
            vocabulary = voc['vocab']
            vocabulary_inv = voc['vocab_inv']
            return vocabulary, vocabulary_inv
    else:
        assert 'vocabulary is not existed'


"""
텍스트로 된 문장을 vocab을 기준으로 
숫자로 변환한다.
"""


def convert_data(sentences, vocabulary):
    UNKNOWN_TAG = '<UNKNOWN/>'

    return np.array(
        [[vocabulary[word] if word in vocabulary else vocabulary[UNKNOWN_TAG] for word in sentence] for sentence in
         sentences])


def inv_convert(sentences, vocabulary_inv):
    return np.array([[vocabulary_inv[idx] for idx in sentence] for sentence in sentences])


"""
학습 데이터를 전처리한다.
"""


def prepare_train_set(pos_path, neg_path, vocab_path, d_type):
    with open(pos_path, 'r', encoding='utf8') as pos:
        positive_examples = list(pos.readlines())
        positive_examples = [s.strip() for s in positive_examples]
        print('positive processing...')
        positive_examples = clean_sentences(positive_examples, d_type)
    with open(neg_path, 'r', encoding='utf8') as neg:
        negative_examples = list(neg.readlines())
        negative_examples = [s.strip() for s in negative_examples]
        print('negative processing...')
        negative_examples = clean_sentences(negative_examples, d_type)

    vocabulary, vocabulary_inv = load_vocab(vocab_path)
    x = positive_examples + negative_examples
    x = convert_data(x, vocabulary)

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    y = np.array(y)

    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv)}
    y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv

