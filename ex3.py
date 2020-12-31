import sys

dev_set_filename = ''
test_set_filename = ''


def get_params():
    global dev_set_filename
    global test_set_filename

    if len(sys.argv) >= 1:
        dev_set_filename = sys.argv[1]
        test_set_filename = sys.argv[2]


# start
def start():

    # get params
    get_params()


# start programs
start()
