import numpy as np


class ProbabilisticGenerative:

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # Assume the training point are sampled from a Gaussian distribution
        # Find the Gaussian distribution behind them
        dimension = x_train.shape[1] - 1

        x_train_1 = np.array([x_train[i][:-1] for i in range(x_train.shape[0]) if y_train[i][0] == 1])
        x_train_0 = np.array([x_train[i][:-1] for i in range(x_train.shape[0]) if y_train[i][0] == 0])

        count_1 = x_train_1.shape[0]
        count_0 = x_train_0.shape[0]

        mean_1 = np.mean(x_train_1, axis=0)
        mean_0 = np.mean(x_train_0, axis=0)

        covariance_1 = np.zeros((dimension, dimension))
        covariance_0 = np.zeros((dimension, dimension))

        for x in x_train_1:
            covariance_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / count_1
        for x in x_train_0:
            covariance_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / count_0

        covariance = (count_1 * covariance_1 + count_0 * covariance_0) / (count_1 + count_0)

        # Compute inverse of covariance matrix.
        # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
        # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
        u, s, v = np.linalg.svd(covariance, full_matrices=False)
        inv_covariance = np.matmul(v.T * 1 / s, u.T)

        weight = np.dot(inv_covariance, mean_0 - mean_1)
        bias = (-0.5) * np.dot(mean_0, np.dot(inv_covariance, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv_covariance, mean_1))\
                + np.log(float(count_0) / count_1)
        self.w = np.append(weight, bias).reshape(-1, 1)

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        # A safe implementation to avoid overflow
        return 0.5 * (1 + np.tanh(0.5 * x))

    def __f(self, x: np.ndarray) -> np.ndarray:
        return self.__sigmoid(np.dot(x, self.w))

    def predict(self, x: np.ndarray) -> np.ndarray:
        # p = P(y=0|x) = sigmoid(wx+b)
        # round(p) = 0 -> y = 1     round(p) = 1 -> y = 0
        return 1 - np.round(self.__f(x)).astype(int)

    def score(self, x_test, y_test):
        return np.mean(y_test == self.predict(x_test))