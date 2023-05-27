"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score

if __name__ == '__main__':
    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(
        X, y, test_size=0.2, shuffle=True)

    # training
    # params = dict()
    params = {
        'learning_rate': 0.1,
        'n_iters': 10000
    }
    print('Training Logistic Regression with Bagging')
    print('Parameters: ', params)
    print('Number of estimators: 9')
    print('Number of samples: ', len(X_train))
    print('Number of features: ', len(X_train.columns))
    print('Number of classes: ', len(y_train.unique()))
    print('Number of iterations: ', params['n_iters'])
    print('Learning rate: ', params['learning_rate'])
    print('Shuffle: True')
    print('Test size: 0.2')
    print('----------------------------------------')
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(
        base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
