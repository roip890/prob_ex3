import sys
from document import Document

# files names
dev_set_filename = ''
test_set_filename = ''
topics_filename = ''

# topics
topics = []
clusters = []


def get_params():
    global dev_set_filename
    global test_set_filename
    global topics_filename
    if len(sys.argv) >= 3:
        dev_set_filename = sys.argv[1]
        test_set_filename = sys.argv[2]
        topics_filename = sys.argv[3]


def topics_data_processing():
    global topics_filename
    global topics
    with open(topics_filename, 'r') as topics_file:
        topics_file_lines = topics_file.readlines()
        for i in range(0, len(topics_file_lines), 2):
            topic_data = topics_file_lines[i:i + 2]
            topic = topic_data[0].strip()
            topics.append(topic)


def dev_data_processing():
    global dev_set_filename
    global clusters
    global topics

    clusters = [[] for i in range(0, 10)]
    with open(dev_set_filename, 'r') as dev_set_file:
        dev_set_file_lines = dev_set_file.readlines()
        for i in range(0, len(dev_set_file_lines), 4):
            cluster_index = int((i / 4) % 9)
            document_data = dev_set_file_lines[i:i + 4]
            document_train_index, document_train_topics = document_train_data_processing(document_data[0])
            document_text = document_data[2].strip()
            document = Document(document_text, document_train_index, document_train_topics, topics)
            clusters[cluster_index].append(document)

def document_train_data_processing(document_train_line):
    document_train_line = document_train_line.strip()
    document_train_line = document_train_line[1:len(document_train_line)-1]
    document_train_data = document_train_line.split('\t')
    document_train_index = document_train_data[1]
    document_train_topics = document_train_data[2:] if len(document_train_data) > 3 else []
    return document_train_index, document_train_topics



# start
def start():
    global clusters
    # get params
    get_params()

    # topics data processing
    topics_data_processing()

    # dev data processing
    dev_data_processing()


# start programs
start()
