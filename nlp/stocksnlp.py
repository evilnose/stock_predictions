# NLP
import string
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Data manipulation
import pandas as pd
import os

# General
import numpy as np

from definitions import root

DATA_DIR = './data/'
WORD2VEC_PATH = root(DATA_DIR, 'GoogleNews-vectors-negative300.bin')


def load_symbols(fpaths):
    df = pd.DataFrame()
    for fpath in fpaths:
        df = pd.concat([df, pd.read_csv(fpath)], ignore_index=True)
    return df


class Word2Vec:
    STOPWORDS = set(stopwords.words('english'))
    STOCKS = load_symbols([root(DATA_DIR, 'NASDAQ.txt'), root(DATA_DIR, 'NYSE.txt')])

    def __init__(self, limit=500000):
        print("Loading word2vec model...")
        self.model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True, limit=limit)
        print("Done.")

    def sentence_to_embedding(self, sentence):
        """
        :param sentence: the sentence to convert to embeddings
        :return: a list of word embedding vectors
        """
        sentence = sentence.translate(string.punctuation)
        emb = []
        for word in sentence.split():
            if self.is_meaningful(word):
                emb.append(self.model.wv[word].tolist())
        return emb

    def top_related_stocks(self, sentence, n=5):
        """
        Return the top-n related stocks given a sentence.
        :param sentence: a sentence.
        :param n: the top number of stocks.
        :return: a numpy array of the symbols of the top-n related stocks.
        """
        words = word_tokenize(sentence)
        sums = Word2Vec.STOCKS.apply(lambda a: self.stock_similarity(a, words), axis=1, result_type='reduce')
        ind = np.argpartition(sums.values, -n)[-n:]
        return Word2Vec.STOCKS.reindex(index=ind)

    def stock_similarity(self, aliases, words):
        """
        Return the average similarity between a stock and a sentence (list of words) by taking the
        average of the similarities of each alias. Aliases that are not in the corpus are omitted, and
        if no alias of a stock is in the corpus, 0 is returned.
        :param aliases: an iterable of all the aliases of a stock.
        :param words: the list of words to compare for similarity.
        :return: the average similarity between the aliases and the words.
        """
        n_aliases = 0
        total_sim = 0
        for alias in aliases:
            if self.is_available(alias):
                n_aliases += 1
                total_sim += self.similarity_sum(alias, words)

        return 0 if n_aliases == 0 else total_sim / n_aliases

    def similarity_sum(self, keyword, words):
        """
        Return the similarity between a keyword and a list of words.
        """
        return sum(self.model.wv.similarity(keyword, w) for w in words)

    def is_meaningful(self, word):
        """
        Only return True when the word is both in the corpus and not a stopword; False otherwise.
        """
        return self.is_available(word) and (word not in Word2Vec.STOPWORDS)

    def is_available(self, word):
        """
        Return True if the word is in the corpus, and False otherwise.
        """
        try:
            _ = self.model[word]
            return True
        except KeyError:
            return False


if __name__ == '__main__':
    wv = Word2Vec()
