import logging
import os
import sys
import time


def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

