import functools
import logging
import os
import random
import re
import unicodedata
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
import numpy as py
import pandas as pd

PAD_TOKEN = '<PAD>'
WHITESPACE_REGEXP = re.compile(r'\s+')

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold

#this function seems to be key.
#Creates a dictionary containing word_id, entity_id, prior_prob and label features for each of train, dev and test set.
def generate_features(dataset, tokenizer, entity_linker, min_count, max_word_length, max_entity_length):

    @functools.lru_cache(maxsize=None)
    def tokenize(text):
        return tokenizer.tokenize(text)

    @functools.lru_cache(maxsize=None)
    def detect_mentions(text):
        return entity_linker.detect_mentions(text)

    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[:len(source_sequence)] = source_sequence
        return ret

    logger.info('Creating vocabulary...')
    word_counter = Counter()
    entity_counter = Counter()
    for instance in tqdm(dataset):
        word_counter.update(t.text for t in tokenize(instance.text))
        entity_counter.update(m.title for m in detect_mentions(instance.text))

    #creates word count dictionary
    words = [word for word, count in word_counter.items() if count >= min_count]
    word_vocab = {word: index for index, word in enumerate(words, 1)}
    word_vocab[PAD_TOKEN] = 0

    #creates entity count dictionary
    entity_titles = [title for title, count in entity_counter.items() if count >= min_count]
    entity_vocab = {title: index for index, title in enumerate(entity_titles, 1)}
    entity_vocab[PAD_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], word_vocab=word_vocab, entity_vocab=entity_vocab)

    for fold in ('train', 'dev', 'test'): #A validation dataset is a dataset of examples used to tune the hyperparameters. It is sometimes also called the development set or the "dev set".
        for instance in dataset.get_instances(fold):
            word_ids = [word_vocab[token.text] for token in tokenize(instance.text) if token.text in word_vocab] #all possible word ids
            entity_ids = []
            prior_probs = []
            for mention in detect_mentions(instance.text):
                if mention.title in entity_vocab: #why mention.title? is there mention.context?
                    # print(mention.title)
                    entity_ids.append(entity_vocab[mention.title]) #appends the context?
                    # print(entity_vocab[mention.title])
                    prior_probs.append(mention.prior_prob)

            ret[fold].append(dict(word_ids=create_numpy_sequence(word_ids, max_word_length, np.int),
                                  entity_ids=create_numpy_sequence(entity_ids, max_entity_length, np.int),
                                  prior_probs=create_numpy_sequence(prior_probs, max_entity_length, np.float32),
                                  label=instance.label))

    return ret


def load_20ng_dataset(dev_size=0.05):
    train_data = []
    test_data = []

    for fold in ('train', 'test'):
        dataset_obj = fetch_20newsgroups(subset=fold, shuffle=False)

        for text, label in zip(dataset_obj['data'], dataset_obj['target']):
            text = normalize_text(text)
            if fold == 'train':
                train_data.append((text, label))
            else:
                test_data.append((text, label))

    dev_size = int(len(train_data) * dev_size)
    random.shuffle(train_data)

    instances = []
    instances += [DatasetInstance(text, label, 'dev') for text, label in train_data[-dev_size:]]
    instances += [DatasetInstance(text, label, 'train') for text, label in train_data[:-dev_size]]
    instances += [DatasetInstance(text, label, 'test') for text, label in test_data]

    return Dataset('20ng', instances, fetch_20newsgroups()['target_names'])


def load_r8_dataset(dataset_path, dev_size=0.05):
    label_names = ['grain', 'earn', 'interest', 'acq', 'trade', 'crude', 'ship', 'money-fx']
    label_index = {t: i for i, t in enumerate(label_names)}

    train_data = []
    test_data = []

    for file_name in sorted(os.listdir(dataset_path)):
        if file_name.endswith('.sgm'): #这是数据的格式？
            with open(os.path.join(dataset_path, file_name), encoding='ISO-8859-1') as f:
                for node in BeautifulSoup(f.read(), 'html.parser').find_all('reuters'): #What does beautiful soup do?
                    text = normalize_text(node.find('text').text)
                    label_nodes = [n.text for n in node.topics.find_all('d')]
                    if len(label_nodes) != 1:
                        continue

                    labels = [label_index[l] for l in label_nodes if l in label_index]
                    if len(labels) == 1:
                        if node['topics'] != 'YES':
                            continue
                        if node['lewissplit'] == 'TRAIN':
                            train_data.append((text, labels[0]))
                        elif node['lewissplit'] == 'TEST':
                            test_data.append((text, labels[0]))
                        else:
                            continue

    dev_size = int(len(train_data) * dev_size)
    random.shuffle(train_data)

    instances = []
    instances += [DatasetInstance(text, label, 'dev') for text, label in train_data[-dev_size:]]
    instances += [DatasetInstance(text, label, 'train') for text, label in train_data[:-dev_size]]
    instances += [DatasetInstance(text, label, 'test') for text, label in test_data]

    return Dataset('r8', instances, label_names)

def load_agnews_dataset(agnews_path, dev_ratio = 0.333, sample_ratio = 0.3):
    def load_data(filename, data_path = agnews_path):
      data_df = pd.read_csv(os.path.join(data_path, filename), header=None)
      data_df.columns = ['rating', 'topic','description']
      data_df.rating = data_df.rating - 1 #map to (0, ..., number of labels)
      data = data_df[['description','rating']].to_numpy()
      return data

    train_data = load_data('train.csv')
    train_data = random.choices(train_data, k = round(sample_ratio*train_data.shape[0]) )
    test_data = load_data('test.csv')

    #train dev split
    dev_size = round(len(train_data) * dev_ratio)
    print(dev_size)
    random.shuffle(train_data)
    dev_data = train_data[:dev_size]
    train_data = train_data[dev_size:]

    instances = []
    instances += [DatasetInstance(data[0], data[1], 'train') for data in train_data]
    instances += [DatasetInstance(data[0], data[1], 'dev') for data in dev_data]
    instances += [DatasetInstance(data[0], data[1], 'test') for data in test_data]
    labels = ['1','2','3','4']
    return Dataset('agnews', instances, labels)


def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, ' ', text)

    # remove accents: https://stackoverflow.com/a/518232
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)

    return text
