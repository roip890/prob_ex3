from document import Document
import numpy as np


class Data(object):
    def __init__(self):
        self.topics = []
        self.documents = []
        self.clusters = []
        self.matrix = None

    def process_data(self, dev_set_filename, test_set_filename, topics_filename):
        # topics data processing
        self.topics_data_processing(topics_filename)

        # dev data processing
        self.dev_data_processing(test_set_filename)

        # init matrix
        self.init_matrix()

    def topics_data_processing(self, topics_filename):
        self.topics = []
        with open(topics_filename, 'r') as topics_file:
            topics_file_lines = topics_file.readlines()
            for i in range(0, len(topics_file_lines), 2):
                topic_data = topics_file_lines[i:i + 2]
                topic = topic_data[0].strip()
                self.topics.append(topic)

    def dev_data_processing(self, dev_set_filename):
        self.clusters = [[] for i in range(0, 10)]
        with open(dev_set_filename, 'r') as dev_set_file:
            dev_set_file_lines = dev_set_file.readlines()
            for i in range(0, len(dev_set_file_lines), 4):
                cluster_index = int((i / 4) % 9)
                document_data = dev_set_file_lines[i:i + 4]
                document_train_index, document_train_topics = self.document_train_data_processing(document_data[0])
                document_text = document_data[2].strip()
                document = Document(document_text, document_train_index, document_train_topics, cluster_index, len(self.topics))
                self.documents.append(document)
                self.clusters[cluster_index].append(document)

    def document_train_data_processing(self, document_train_line):
        document_train_line = document_train_line.strip()
        document_train_line = document_train_line[1:len(document_train_line) - 1]
        document_train_data = document_train_line.split('\t')
        document_train_index = document_train_data[1]
        document_train_topics = document_train_data[2:] if len(document_train_data) > 3 else []
        return document_train_index, document_train_topics

    def init_matrix(self):
        self.matrix = np.empty(shape=(len(self.topics), len(self.documents)))
