import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from definitions import root
from nlp.stocksnlp import Word2Vec
from model.common import WINDOW_TIME_STEPS, MODEL_TRAINED_PATH, EMB_SIZE, HEADLINE_SEQ_LEN
from model.build_model import get_overall_model

from sklearn.model_selection import train_test_split

# Hyperparameters
HIDDEN_SIZE_HEADLINE = 50  # Size of LSTM hidden layer for headline condenser
HIDDEN_SIZE_MAIN = 10  # Size of LSTM hidden layer for main
N_CONDENSED_OUT = 3  # Output feature dimensions for the condenser
# Input feature dimensions of the numerical data (only price for now)
N_NUM = 1
N_BATCH = 100  # Training batch size
N_EPOCHS = 30
VAL_SPLIT = 0.33

HEADLINE_FILE = root('./data/apple_headlines.csv')
PRICE_FILE = root('./data/apple_prices.csv')


def preprocessing(headline_file=HEADLINE_FILE, price_file=PRICE_FILE, window=WINDOW_TIME_STEPS, n_days_forward=7,
                  wv=None):
    """
    Reads the apple prices and headlines csv and combines their data into one pandas DataFrame
    :return: a DataFrame containing headlines and prices, indexed by date.
    """
    '''Unrelated initialization of condenser (not needed if we're not saving model architecture)'''
    # global headline_condenser_g
    # headline_condenser_g = make_condenser(in_shape=(HEADLINE_SEQ_LEN, EMB_SIZE))

    # Read data
    sentiments = pd.read_csv(
        headline_file, encoding='utf-8', parse_dates=['date'])
    prices = pd.read_csv(price_file, encoding='utf-8', parse_dates=['date'])

    # Preprocess data
    if not wv:
        wv = Word2Vec()
    sentiments['headline'] = sentiments['headline'].apply(wv.sentence_to_embedding)
    sentiments['headline'], time_steps = zero_pad(sentiments['headline'])
    merged = pd.merge(prices, sentiments, on="date", how="left")

    # Replace NaN
    zero_col = np.zeros(shape=(len(merged), time_steps, EMB_SIZE))
    d = pd.Series(zero_col.tolist())
    merged['headline'] = merged['headline'].fillna(d)
    merged['headline'] = merged['headline'].apply(lambda h: np.asarray(h, dtype=np.float64))
    return package_data(np.asarray(merged['headline'].tolist()), np.asarray(merged['price'].tolist()), window,
                        n_days_forward)


def get_packaged_ind(m, window, n_days_forward):
    """
    Return an intermediate index array for the time series characterized by the parameters. Note that both the training
    and the prediction data come from the same series.
    :param m: Number of training examples.
    :param window: Number of time-steps in each training example.
    :param n_days_forward: Forward offset for the prediction data. E.g. if n_days_forward == 5, then for each training
    example, the prediction time-step is 5 time-steps later than the last training time-step.
    :return: (x, y), list of indices by which the training and testing examples should be chosen, respectively. Note
    that x is a 2d-list since each training example would have multiple time-steps.
    """
    if n_days_forward < 1:
        raise ValueError("n_days_forward should be greater or equal to 1.")
    # currently data (x) is overlapping
    y = list(range(window + n_days_forward - 1, m))
    x = list(list(range(i - n_days_forward - window + 1, i - n_days_forward + 1)) for i in y)
    return x, y


def from_2d_ind(data, ind):
    """
    Reindex the data based on given indices
    :param data: Time-series data.
    :param ind: Indices processed from get_packaged_ind.
    :return: Reindexed data.
    """
    processed = []
    for ex_ind in ind:  # Get index for each example
        processed.append(data[ex_ind])
    return np.asarray(processed)


def package_data(headlines, prices, window, n_days_forward=7):
    """
    Packages (e.g. splits into training examples) both the headlines and the price series and returns
    the processed series in a list
    :param headlines: A series of headlines data.
    :param prices: A series of price data.
    :param window: Number of time steps for each training example.
    :param n_days_forward: An integer that indicates how many days in the future is each prediction relative to the last
    day in each training example.
    :return: A list of processed series [headline, price_x, price_y].
    """
    # headlines = headlines.as_matrix()  # TODO convert this to ndarray
    # prices = np.asarray(prices, dtype=np.float)
    x_ind, y_ind = get_packaged_ind(len(headlines), window, n_days_forward)
    price_x = from_2d_ind(prices, x_ind)
    price_y = prices[y_ind]
    headline_in = from_2d_ind(headlines, x_ind)
    m = len(y_ind)
    return price_x.reshape(m, window, 1), price_y.reshape(m, 1), headline_in.squeeze()


def zero_pad(series):
    """
    Given a list of list of vectors, zero-pad/truncate so that each list of vectors is
    of the same length, which defaults to the maximum list length.
    :param series: the series of embeddings to zero pad
    :return: the series, except every embedding vector within is zero padded to matching length
    """
    max_length = max(len(x) for x in series)
    copy = np.asarray(series.copy())
    for i in range(copy.shape[0]):
        s = copy[i]
        if len(s) != max_length:
            copy[i] += [np.zeros(EMB_SIZE) for _ in range(max_length - len(s))]
    return copy, max_length


def read_json_string(path):
    with open(path, 'r') as f:
        return f.read()


def plot_hist(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'cross_val'], loc='upper left')
    plt.show()


def get_stratify_list(y_sorted, n_bins):
    y_sorted = y_sorted.reshape(len(y_sorted), 1)
    bins = np.linspace(y_sorted[0], y_sorted[-1], n_bins)
    return np.digitize(y_sorted, bins, right=True)


def stratified_split_ind(y, test_size, n_bins, rand_state=0):
    sorted_ind = np.argsort(y, axis=0)
    return train_test_split(np.arange(len(y))[sorted_ind], test_size=test_size,
                            stratify=get_stratify_list(y[sorted_ind], n_bins),
                            random_state=rand_state)


def split_train_test_val(y, test_size=0.2, val_size=0.3, n_bins=20):
    """Splits given data into train, test, and validation sets, each with proportions given by the parameters.
    Note that validation set is split from the train set and thus has an overall proportion of test_size * val_size"""
    train_and_val_ind, test_ind = stratified_split_ind(y, test_size, n_bins)
    train_ind_t, val_ind_t = stratified_split_ind(y[train_and_val_ind], val_size, n_bins)
    return np.asarray(train_and_val_ind[train_ind_t]).squeeze(), np.asarray(
        train_and_val_ind[val_ind_t]).squeeze(), np.asarray(test_ind).squeeze()


if __name__ == "__main__":
    word2vec = Word2Vec()
    price_x, price_y, headline = preprocessing(n_days_forward=31, wv=word2vec)
    train, test, val = split_train_test_val(price_y)
