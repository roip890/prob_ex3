from data import Data
from document import Document
import math
import numpy as np
import time

class ExpectationMaximizationAlgorithm(object):

    def __init__(self, data):
        # algorithm params
        self.data = data
        self.alphas = [0 for i in range(0, len(data.clusters))]

        print('start', time.time() - self.data.start_time)
        print('expectation - start', time.time() - self.data.start_time)
        self.expectation()
        print('expectation - end', time.time() - self.data.start_time)
        print('maximization - start', time.time() - self.data.start_time)
        self.maximization()
        print('maximization - end', time.time() - self.data.start_time)
        print('expectation - start', time.time() - self.data.start_time)
        self.expectation()
        print('expectation - end', time.time() - self.data.start_time)
        print('maximization - start', time.time() - self.data.start_time)
        self.maximization()
        print('maximization - end', time.time() - self.data.start_time)

    # expectation step
    def expectation(self):
        self.normalize_alpha()
        for t in range(0, len(self.data.documents)):
            # update z_i for this document
            self.data.z[t] = np.array([self.calculate_z_i(t, j) for j in range(len(self.data.clusters))])
            m = np.max(self.data.z[t])
            self.data.z[t] = np.array([0 if z_i - m < self.data.k else z_i - m for z_i in self.data.z[t]])
            w_t_i_denominator = sum([math.pow(math.e, z_i) for z_i in self.data.z[t] if z_i != 0])
            for i in range(0, len(self.data.clusters)):
                self.data.w[t][i] = 0 if self.data.z[t][i] == 0 else math.pow(math.e, self.data.z[t][i]) / w_t_i_denominator

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

    def normalize_alpha(self):
        new_alphas = [max(alpha, self.data.eps) for alpha in self.data.a]
        alphas_sum = sum(new_alphas)
        self.data.a = [alpha / alphas_sum for alpha in new_alphas]

    def calculate_z_i(self, t, i):
        ln_alpha_i = math.log(self.data.a[i])
        p_i_k_sum = np.sum([math.log(self.data.p[i][self.data.v_i[k]]) * self.data.n[t][self.data.v_i[k]] for k in self.data.documents[t].words_set])
        return ln_alpha_i + p_i_k_sum
