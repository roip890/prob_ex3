#Itay Yair 308204239

from document import Document
import numpy as np
import math
import time
from collections import Counter
from operator import add
from functools import reduce


class Data(object):
    def __init__(self):

        # calculate start time for debugging
        self.start_time = time.time()

        # topics
        self.topics = []
        self.topics_dict = {}

        # documents
        self.documents = []

        # cluster
        self.clusters = []

        # vocabulary
        self.v = []
        self.v_dict = {}
        self.v_i = {}

        # max k
        self.max_k = -math.inf

        # matrices
        self.p = []
        self.n = []
        self.n_t = []
        self.w = []
        self.a = []
        self.z = []
        self.z_m = []
        self.k = 10
        self.lamb = 0.99
        self.eps = math.pow(math.e, -8)

    def process_data(self, dev_set_filename, test_set_filename, topics_filename):
        # topics data processing
        self.topics_data_processing(topics_filename)

        # dev data processing
        self.dev_data_processing(dev_set_filename)

        # init matrices
        self.init_matrices()

    def topics_data_processing(self, topics_filename):
        self.topics = []
        with open(topics_filename, 'r') as topics_file:
            topics_file_lines = topics_file.readlines()
            for i in range(0, len(topics_file_lines), 2):
                topic_data = topics_file_lines[i:i + 2]
                topic = topic_data[0].strip()
                self.topics.append(topic)
            self.topics_dict = {self.topics[topic_index]: topic_index for topic_index in range(0, len(self.topics))}

    def dev_data_processing(self, dev_set_filename):
        self.clusters = [[] for i in range(0, len(self.topics))]
        self.v = []
        with open(dev_set_filename, 'r') as dev_set_file:
            dev_set_file_lines = dev_set_file.readlines()
            for i in range(0, len(dev_set_file_lines), 4):
                cluster_index = int((i / 4) % 9)
                document_data = dev_set_file_lines[i:i + 4]
                document_train_index, document_train_topics = self.document_train_data_processing(document_data[0])
                document_train_topics_index = [self.topics_dict[topic] for topic in document_train_topics if topic in self.topics]
                document_text = document_data[2].strip()
                document = Document(document_text, document_train_index, document_train_topics, document_train_topics_index, cluster_index, len(self.topics))
                self.max_k = max(self.max_k, len(document.words_set))
                self.v.extend(document.words_set)
                self.documents.append(document)
                self.clusters[cluster_index].append(document)
                self.v_dict = Counter(self.v_dict) + Counter(document.words_count_dict)
        self.v_dict = {k: v for k, v in self.v_dict.items() if v > 3}
        # self.v = list(set(self.v))
        self.v = list(self.v_dict.keys())
        self.v_i = {self.v[i]: i for i in range(0, len(self.v))}

    # process train row per document
    def document_train_data_processing(self, document_train_line):
        document_train_line = document_train_line.strip()
        document_train_line = document_train_line[1:len(document_train_line) - 1]
        document_train_data = document_train_line.split('\t')
        document_train_index = document_train_data[1]
        document_train_topics = document_train_data[2:] if len(document_train_data) > 2 else []
        return document_train_index, document_train_topics

    # init all matrices of the algorithm data
    def init_matrices(self):

        # alpha
        self.a = np.zeros(shape=len(self.clusters))
        for i in range(0, len(self.clusters)):
            self.a[i] = 1 / len(self.clusters)

        # z
        self.z = np.ones(shape=(len(self.documents), len(self.clusters)))
        self.z_m = np.ones(shape=(len(self.documents), 1))

        # n
        self.n = np.zeros(shape=(len(self.documents), len(self.v)))
        for document_index in range(0, len(self.documents)):
            document = self.documents[document_index]
            for word in document.words_set:
                if word in self.v:
                    self.n[document_index][self.v_i[word]] = document.words_count_dict.get(word, 0)

        # nt
        self.n_t = self.n.sum(axis=1)

        self.p = np.ones(shape=(len(self.clusters), len(self.v)))

        # w
        self.w = np.zeros(shape=(len(self.documents), len(self.clusters)))
        for document_index in range(0, len(self.documents)):
            for cluster_index in range(0, len(self.clusters)):
                if self.documents[document_index].cluster_index == cluster_index:
                    self.w[document_index][cluster_index] = 2 / (len(self.clusters) + 1)
                else:
                    self.w[document_index][cluster_index] = 1 / (len(self.clusters) + 1)
