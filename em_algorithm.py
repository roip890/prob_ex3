from data import Data
from document import Document
import math
import numpy as np
import time
import matplotlib.pyplot as plt


class ExpectationMaximizationAlgorithm(object):

    def __init__(self, data):
        # algorithm params
        self.data = data
        self.likelihood_values = []
        self.perplexity_values = []

    def start_algorithm(self):
        # print('start', time.time() - self.data.start_time)
        for i in range(0, 20):
            self.maximization()
            self.expectation()
        self.plot(self.likelihood_values[1:], [i for i in range(0, len(self.likelihood_values[1:]))], 'likelihood', 'iterations')
        # self.plot(self.perplexity_values[1:], [i for i in range(0, len(self.perplexity_values[1:]))], 'perplexity', 'iterations')

    # expectation step
    def expectation(self):
        self.normalize_alpha()
        likelihood = 0
        for t in range(0, len(self.data.documents)):
            # update z_i for this document
            self.data.z[t] = np.array([self.calculate_z_i(t, j) for j in range(len(self.data.clusters))])
            self.data.z_m[t] = np.max(self.data.z[t])
            z_i_minus_m = self.data.z[t] - self.data.z_m[t]
            w_t_i_denominator = sum([math.pow(math.e, z_i) for z_i in z_i_minus_m if z_i > - self.data.k])
            for i in range(0, len(self.data.clusters)):
                self.data.w[t][i] = 0 if z_i_minus_m[i] < - self.data.k else (math.pow(math.e, z_i_minus_m[i]) / w_t_i_denominator)
            likelihood += self.data.z_m[t][0] + math.log(w_t_i_denominator)
        print('likelihood', likelihood)
        self.likelihood_values.append(likelihood)
        self.perplexity_values.append(math.pow(2, (-1/sum(self.data.n_t)) * likelihood))

    # maximization step
    def maximization(self):
        n = len(self.data.documents)
        lamb = self.data.lamb
        lamb_v = self.data.lamb * len(self.data.v)
        for i in range(0, len(self.data.clusters)):
            self.data.a[i] = self.calculate_alpha_i(i, n)
            p_numerator = self.data.w.T.dot(self.data.n) + lamb
            p_denominator = self.data.w.T.dot(self.data.n_t) + lamb_v
            # p_denominator = np.array([p_denominator, ] * len(self.data.v)).T
            self.data.p = (p_numerator.T / p_denominator).T

    def calculate_alpha_i(self, i, n):
        return 1 / n * (sum(self.data.w[:][i]))

    def normalize_alpha(self):
        new_alphas = [max(alpha, self.data.eps) for alpha in self.data.a]
        alphas_sum = sum(new_alphas)
        self.data.a = [alpha / alphas_sum for alpha in new_alphas]

    def calculate_z_i(self, t, i):
        ln_alpha_i = math.log(self.data.a[i])
        p_i_k_sum = np.sum([math.log(self.data.p[i][self.data.v_i[k]]) * self.data.n[t][self.data.v_i[k]] for k in self.data.documents[t].words_set])
        return ln_alpha_i + p_i_k_sum

    def plot(self, y_values, x_values, y_label, x_label):
        plt.plot(x_values, y_values)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()
