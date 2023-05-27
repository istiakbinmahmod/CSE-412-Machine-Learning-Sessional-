import pandas as pd
from random import randrange

def standardize(data):
    """
    function for standardizing the data
    :param data:
    :return:
    """
    data_x = data[data.columns[:-1]]
    data[data.columns[:-1]] = (data_x-data_x.mean())/data_x.std()

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv('data_banknote_authentication.csv')
    standardize(data)

    # sepatare the data into features and labels
    features = data.drop(columns='isoriginal', axis=1)
    target = data['isoriginal']

    return features, target


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    df = pd.concat([X, y], axis=1)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop(columns='isoriginal', axis=1)
    y = df['isoriginal']

    X_train = X[:int(len(X) * (1 - test_size))]
    y_train = y[:int(len(y) * (1 - test_size))]
    X_test = X[int(len(X) * (1 - test_size)):]
    y_test = y[int(len(y) * (1 - test_size)):]

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    dataset = pd.concat([X, y], axis=1)
    new_dataset = list()
    n_sample = len(dataset)

    while len(new_dataset) < n_sample:
        index = randrange(len(dataset))
        new_dataset.append(dataset.iloc[index])

    new_dataset = pd.DataFrame(new_dataset)
    X_sample = new_dataset.drop(columns='isoriginal', axis=1)
    y_sample = new_dataset['isoriginal']

    return X_sample, y_sample
