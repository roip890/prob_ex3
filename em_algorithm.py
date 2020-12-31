from data import Data
from document import Document
import math


class ExpectationMaximizationAlgorithm(object):

    def __init__(self, data):
        # algorithm params
        self.data = data
        self.alphas = [0 for i in range(0, len(data.clusters))]

        self.maximization()
        self.expectation()
        self.maximization()

    def expectation(self):
        for document in self.data.documents:
            w_t_i_denominator = self.calculate_w_t_i_denominator(document)
            for cluster_index in range(0, len(self.data.clusters)):
                w_t_i_numerator = self.calculate_w_t_i_numerator(cluster_index, document)
                document.w[cluster_index] = w_t_i_numerator / w_t_i_denominator

    def maximization(self):
        n = len(self.data.documents)
        for cluster_index in range(0, len(self.data.clusters)):
            self.alphas[cluster_index] = self.calculate_alpha_i(cluster_index, n)
            for document in self.data.documents:
                for word in document.words_set:
                    document.p[word] = self.calculate_p_i_k(cluster_index, word)

    def calculate_alpha_i(self, cluster_index, n):
        return 1 / n * (sum([document.w[cluster_index] for document in self.data.documents]))

    def calculate_p_i_k(self, cluster_index, word):
        numerator = 0
        denominator = 0
        for document in self.data.documents:
            numerator += document.w[cluster_index] * document.n.get(word, 0)
            denominator += document.w[cluster_index] * document.word_count
        return numerator / denominator

    def calculate_w_t_i_numerator(self, cluster_index, document):
        return self.calculate_alpha_p_i_prod(cluster_index, document)

    def calculate_w_t_i_denominator(self, document):
        return sum(self.calculate_alpha_p_i_prod(j, document) for j in range(0, len(self.data.clusters)))

    def calculate_alpha_p_i_prod(self, cluster_index, document):
        alpha_i = self.alphas[cluster_index]
        p_i_prod = math.prod(self.calculate_p_i_k(cluster_index, word) for word in document.words_set)
        return alpha_i * p_i_prod
