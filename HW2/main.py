import pickle
import gzip
import numpy as np
import urllib.request
import json
from numpy.linalg import inv
import pandas as pd
from sklearn.preprocessing import scale

data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


