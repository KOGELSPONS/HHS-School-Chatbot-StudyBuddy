import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import gensim.downloader as api
import spacy
from collections import Counter

### Loading Models ###

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

training_model = api.load("glove-wiki-gigaword-100")

