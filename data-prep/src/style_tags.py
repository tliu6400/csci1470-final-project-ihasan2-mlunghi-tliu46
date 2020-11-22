"""
Generates tags for training the tagger
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from scipy import sparse
from collections import namedtuple
import pickle
import csv
import logging

tqdm.pandas()

tfidf_stats = namedtuple()