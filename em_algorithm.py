from data import Data
from document import Document
import math
import numpy as np
import time

class ExpectationMaximizationAlgorithm(object):

    def __init__(self, data):
        # algorithm params
        self.data = data
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
            self.data.z_m[t] = np.max(self.data.z[t])
            z_i_minus_m = self.data.z[t] - self.data.z_m[t]
            w_t_i_denominator = sum([math.pow(math.e, z_i) for z_i in z_i_minus_m if z_i > - self.data.k])
            for i in range(0, len(self.data.clusters)):
                self.data.w[t][i] = 0 if z_i_minus_m[i] < - self.data.k else math.pow(math.e, z_i_minus_m[i]) / w_t_i_denominator

    # maximization step
    def maximization(self):
        n = len(self.data.documents)
        lamb = self.data.lamb
        lamb_v = self.data.lamb * sum(self.data.v)
        for i in range(0, len(self.data.clusters)):
            self.data.a[i] = self.calculate_alpha_i(i, n)
            p_numerator = self.data.w.T.dot(self.data.n) + lamb
            p_denominator = self.data.w.T.dot(self.data.n_t) + lamb_v
            self.data.p = p_numerator / p_denominator

    def calculate_alpha_i(self, i, n):
        return 1 / n * (sum([self.data.w[t][i] for t in range(0, len(self.data.documents))]))

    def normalize_alpha(self):
        new_alphas = [max(alpha, self.data.eps) for alpha in self.data.a]
        alphas_sum = sum(new_alphas)
        self.data.a = [alpha / alphas_sum for alpha in new_alphas]

    def calculate_z_i(self, t, i):
        ln_alpha_i = math.log(self.data.a[i])
        p_i_k_sum = np.sum([math.log(self.data.p[i][self.data.v_i[k]]) * self.data.n[t][self.data.v_i[k]] for k in self.data.documents[t].words_set])
        return ln_alpha_i + p_i_k_sum

    # def calculate_likelihood(self):
    #     res = 0
    #     for t in range(0, len(self.data.documents)):
    #         m_t =
