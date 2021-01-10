class Document(object):
    def __init__(self, text, index, topics, cluster_index, cluster_count):
        # document params
        self.text = text
        self.index = index
        self.topics = topics
        self.cluster_index = cluster_index

        # init
        self.words = []
        self.words_set = []
        self.word_count = 0
        self.word_set_count = 0
        self.words_count_dict = {}
        self.words_likelihood_dict = {}

        # init
        self.data_process()

    def data_process(self):
        # init
        self.words = []
        self.words_set = []
        self.word_count = 0
        self.word_set_count = 0
        self.words_count_dict = {}

        # calculate word count
        self.words = self.text.split()
        for word in self.words:
            self.words_count_dict[word] = self.words_count_dict.get(word, 0) + 1

        self.words = self.words
        self.word_count = len(self.words)
        self.word_set_count = len(self.words_count_dict.values())
        self.words_set = list(self.words_count_dict.keys())

        # p dict
        for word in self.words_count_dict.keys():
            self.words_likelihood_dict[word] = self.words_count_dict[word] / self.word_count
