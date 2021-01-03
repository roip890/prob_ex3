from data import Data
from document import Document
import math
import numpy as np


class ExpectationMaximizationAlgorithm(object):

    def __init__(self, data):
        # algorithm params
        self.data = data
        self.alphas = [0 for i in range(0, len(data.clusters))]

        print('start')
        print('expectation - start')
        self.expectation()
        print('expectation - end')
        print('maximization - start')
        self.maximization()
        print('maximization - end')
        print('expectation - start')
        self.expectation()
        print('expectation - end')
        print('maximization - start')
        self.maximization()
        print('maximization - end')

    # expectation step
    def expectation(self):
        for t in range(0, len(self.data.documents)):
            w_t_i_denominator = self.calculate_w_t_i_denominator(t)
            for i in range(0, len(self.data.clusters)):
                w_t_i_numerator = self.calculate_w_t_i_numerator(t, i)
                self.data.w[t][i] = w_t_i_numerator / w_t_i_denominator

    def calculate_w_t_i_numerator(self, t, i):
        return self.calculate_alpha_p_i_prod(t, i)

    def calculate_w_t_i_denominator(self, t):
        return sum([self.calculate_alpha_p_i_prod(t, j) for j in range(0, len(self.data.clusters))])

    def calculate_alpha_p_i_prod(self, t, i):
        alpha_i = self.alphas[i]
        p_i_prod = np.prod([math.pow(self.data.p[i][self.data.v_i[k]], self.data.n[t][self.data.v_i[k]]) for k in self.data.documents[t].words_set])
        return alpha_i * p_i_prod

    # maximization step
    def maximization(self):
        n = len(self.data.documents)
        for i in range(0, len(self.data.clusters)):
            self.alphas[i] = self.calculate_alpha_i(i, n)
            p_i_k_denominator = self.calculate_p_i_k_denominator(i)
            for k in self.data.v:
                p_i_k_numerator = self.calculate_p_i_k_numerator(i, k)
                self.data.p[i][self.data.v_i[k]] = p_i_k_numerator / p_i_k_denominator

    def calculate_alpha_i(self, i, n):
        return 1 / n * (sum([self.data.w[t][i] for t in range(0, len(self.data.documents))]))

    def calculate_p_i_k_numerator(self, i, k):
        w_t_arr = self.data.w[:, i]
        n_t_k = self.data.n[:, self.data.v_i[k]]
        return sum(np.multiply(n_t_k, w_t_arr))

    def calculate_p_i_k_denominator(self, i):
        n_t_arr = self.data.n.sum(axis=1)
        w_t_arr = self.data.w[:, i]
        return sum(np.multiply(n_t_arr, w_t_arr))
