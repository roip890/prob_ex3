import sys
from data import Data
from em_algorithm import ExpectationMaximizationAlgorithm
import numpy as np
data = Data()
em_alg = None


def generate_data():
    global data
    if len(sys.argv) >= 3:
        dev_set_filename = sys.argv[1]
        test_set_filename = sys.argv[2]
        topics_filename = sys.argv[3]

        data.process_data(dev_set_filename, test_set_filename, topics_filename)


# start
def start():
    # generate data
    generate_data()

    em_alg = ExpectationMaximizationAlgorithm(data)

# start programs
start()
