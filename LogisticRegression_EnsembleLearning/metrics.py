"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive = np.sum(y_true * y_pred)
    true_negative = np.sum((1 - y_true) * (1 - y_pred))
    pos_neg = np.sum(y_true) + np.sum(1 - y_true)
    acc = (true_positive + true_negative) / pos_neg
    # acc = np.sum(y_true == y_pred) / len(y_true)
    return acc


def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive = np.sum(y_true * y_pred)
    pos = np.sum(y_pred)
    prec = true_positive / pos
    # prec = np.sum(y_true * y_pred) / np.sum(y_pred)
    return prec


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive = np.sum(y_true * y_pred)
    false_negative = np.sum(y_true * (1 - y_pred))
    rec = true_positive / (true_positive + false_negative)
    # rec = np.sum(y_true * y_pred) / np.sum(y_true)
    return rec


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    f1 = 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred) / (
        precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
    # f1 = f1_score(y_true, y_pred)
    return f1
