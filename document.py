class Document(object):
    def __init__(self, text, index, topics, cluster_index, cluster_count):
        # document params
        self.text = text
        self.index = index
        self.topics = topics
        self.cluster_index = cluster_index
        self.words = []
        self.words_set = []
        self.word_count = 0
        self.word_set_count = 0
        self.n = {}
        self.p = {}
        self.w = [1 if i == cluster_index else 0 for i in range(0, cluster_count)]
        self.a = [1 if i == cluster_index else 0 for i in range(0, cluster_count)]

        # init
        self.data_process()

    def data_process(self):
        # calculate word count
        self.words = self.text.split()
        for word in self.words:
            self.n[word] = self.n.get(word, 0) + 1

        self.words = self.words
        self.word_count = len(self.words)
        self.word_set_count = len(self.n.values())
        self.words_set = self.n.keys()

        # p dict
        for word in self.n.keys():
            self.p[word] = self.n[word] / self.word_count
