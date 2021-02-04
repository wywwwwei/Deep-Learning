import argparse
import pandas as pd
import csv
import numpy as np
from model.lr import LinearRegression
from typing import Tuple


def parse_arg() -> Tuple[str, str, str]:
    parser = argparse.ArgumentParser(description="ML2020 Spring HW1")

    parser.add_argument("-t", "--train", metavar="train_set", default="./train.csv", help="the path of the train set")
    parser.add_argument("-i", "--input", metavar="test_set", default="./test.csv", help="the path of the test set")
    parser.add_argument("-o", "--output", metavar="output_file", default="./submit.csv", help="the path of the output file")

    args = parser.parse_args()
    return (args.train, args.input, args.output)


if __name__ == "__main__":
    train_file, test_file, out_file = parse_arg()

    train_data = pd.read_csv(train_file, encoding="big5")
    print(train_data.head())

    # preprocess
    # [12months * 20days * 18observations, 24hours] = [4320,24]
    cols = train_data.columns[3:]
    train_data[cols] = train_data[cols].replace("NR", 0)
    train = train_data[cols].to_numpy()
    print(f"original train set shape: {train.shape}")

    # extract features
    # rearrange by month
    # each month: [18features, 24hours * 20days] = [18,480]
    train_by_month = []
    for month in range(12):
        samples = np.empty([18, 24 * 20])
        for day in range(20):
            new_col = day * 24
            old_row = 18 * (month * 20 + day)
            samples[:, new_col:new_col + 24] = train[old_row:old_row + 18, :]
        train_by_month.append(samples)
    print(f"train set rearranged by month: {len(train_by_month)} * {train_by_month[0].shape}")

    # use previous 9-hours data as features and the 10-th-hour PM2.5 as answer
    # totally (24hours * 20days - 9) * 12months = 471samples/month * 12months = 5652 train samples
    x: np.ndarray = np.empty([5652, 9 * 18], dtype=float)
    y: np.ndarray = np.empty([5652, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                new_pos = month * 471 + day * 24 + hour
                old_pos = day * 24 + hour
                x[new_pos, :] = train_by_month[month][:, old_pos:old_pos + 9].reshape(1, -1)
                y[new_pos, 0] = train_by_month[month][9, old_pos + 9]
    print(f"extracted train x: {x.shape}")

    # normalize
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(x.shape[0]):  # 5652
        for j in range(x.shape[1]):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

    # split
    # add a 1 to handle the constant term
    x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1).astype(float)
    train_x, validate_x = np.split(x, [int(0.8 * x.shape[0])])
    train_y, validate_y = np.split(y, [int(0.8 * y.shape[0])])

    # model
    model = LinearRegression()
    model.fit(train_x, train_y)
    print(f"validation set score: {model.score(validate_x,validate_y,precision=2.0)}")

    # treat the test data the same as the train data
    test_data = pd.read_csv(test_file, header=None, encoding="big5")
    cols = test_data.columns[2:]
    test_data[cols] = test_data[cols].replace("NR", 0)
    test = test_data[cols].to_numpy()
    test_x = np.empty([240, 18 * 9], dtype=float)
    for i in range(240):
        test_x[i, :] = test[i * 18:(i + 1) * 18, :].reshape(1, -1)
    for i in range(240):
        for j in range(test_x.shape[1]):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([test_x.shape[0], 1]), test_x), axis=1).astype(float)
    predict_y = model.predict(test_x)

    # output
    with open(out_file, mode='w', newline='') as sf:
        csv_writer = csv.writer(sf)
        header = ["id", "value"]
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id. ' + str(i), predict_y[i][0]]
            csv_writer.writerow(row)
