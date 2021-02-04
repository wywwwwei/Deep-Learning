import argparse
import csv
import numpy as np
from typing import Tuple
from model.lr import LogisticRegression
from model.pg import ProbabilisticGenerative


def parse_arg() -> Tuple[str, str, str, str]:
    parser = argparse.ArgumentParser(description="ML2020 Spring HW2")

    parser.add_argument("-x", "--xtrain", metavar="train_x", default="./data/X_train", help="the path of the train x")
    parser.add_argument("-y", "--ytrain", metavar="train_y", default="./data/Y_train", help="the path of the train y")
    parser.add_argument("-i", "--input", metavar="test_x", default="./data/X_test", help="the path of the test x")
    parser.add_argument("-o", "--output", metavar="output_file", default="./submit.csv", help="the path of the output file")

    args = parser.parse_args()
    return (args.xtrain, args.ytrain, args.input, args.output)


def normalize(x: np.ndarray, train: bool = True, x_mean=None, x_std=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train:
        x_mean = np.mean(x, axis=0).reshape(1, -1)
        x_std = np.std(x, axis=0).reshape(1, -1)
    x = (x - x_mean) / (x_std + 1e-8)
    return x, x_mean, x_std


if __name__ == "__main__":
    xtrain_file, ytrain_file, xtest_file, out_file = parse_arg()
    np.random.seed(0)

    # Parse csv files to numpy array
    with open(xtrain_file) as f:
        next(f)
        x_train: np.ndarray = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    with open(ytrain_file) as f:
        next(f)
        y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=int)
    with open(xtest_file) as f:
        next(f)
        x_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    print(f"x_train shape: {x_train.shape}")  # [54256, 510]
    print(f"x_test shape: {x_test.shape}")

    # normalize
    x_train, x_mean, x_std = normalize(x_train)  # [1, 510]
    x_test, _, _ = normalize(x_test, train=False, x_mean=x_mean, x_std=x_std)

    # split
    x_train = np.concatenate((x_train, np.ones([x_train.shape[0], 1])), axis=1).astype(float)
    x_test = np.concatenate((x_test, np.ones([x_test.shape[0], 1])), axis=1).astype(float)
    x_train, x_dev = np.split(x_train, [int(0.9 * x_train.shape[0])])
    y_train, y_dev = np.split(y_train, [int(0.9 * y_train.shape[0])])
    print(f"train x size: {x_train.shape}")
    print(f"train y size: {y_train.shape}")
    print(f"dev x size: {x_dev.shape}")

    # model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # score
    print(f"logistic regression model score on development set: {model.score(x_dev,y_dev)}")

    # prediction
    predict = model.predict(x_test)

    # probabilistic generative model
    model_2 = ProbabilisticGenerative()
    model_2.fit(x_train, y_train)
    print(f"probabilistic generative model score on development set: {model_2.score(x_dev, y_dev)}")

    # output
    with open(out_file, mode='w', newline='') as sf:
        csv_writer = csv.writer(sf)
        header = ["id", "label"]
        csv_writer.writerow(header)
        for i in range(240):
            row = [str(i), predict[i][0]]
            csv_writer.writerow(row)
