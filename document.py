class Document(object):
    def __init__(self, text, index, topics, all_topics):
        # document params
        self.text = text
        self.index = index
        self.topics = topics
        self.all_topics = all_topics
        self.words = []
        self.word_count = 0
        self.word_set_count = 0
        self.n = {}
        self.p = {}

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

        # p dict
        for word in self.n.keys():
            self.p[word] = self.n[word] / self.word_count
