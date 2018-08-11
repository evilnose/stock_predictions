import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def root(*paths):
    return os.path.join(*((ROOT_DIR,) + paths))