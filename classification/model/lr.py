from typing import Tuple
import numpy as np
from numpy import random


def _shuffle(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    randomize = np.arange(x_train.shape[0])
    np.random.shuffle(randomize)
    return x_train[randomize], y_train[randomize]


class LogisticRegression:

    def __init__(self) -> None:
        super().__init__()

    def __gradient(self, x_train: np.ndarray, y_train: np.ndarray, y_predict: np.ndarray) -> np.ndarray:
        diff = y_train - y_predict
        grad = -np.dot(x_train.transpose(), diff)
        return grad

    def __cross_entropy(self, y_train: np.ndarray, y_predict: np.ndarray, eps: float = 1e-8) -> float:
        y_predict = np.clip(y_predict, eps, 1.0 - eps)  # avoid log error
        return -np.sum(y_train * np.log(y_predict) + (1 - y_train) * np.log(1 - y_predict))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, lr: float = 0.2, batch_size: int = 8):
        dimension = x_train.shape[1]
        self.w = np.zeros([dimension, 1])
        train_size = x_train.shape[0]

        step = 1
        for epoch in range(epochs):
            x_train, y_train = _shuffle(x_train, y_train)
            for i in range(train_size // batch_size):
                x = x_train[i * batch_size:(i + 1) * batch_size]
                y = y_train[i * batch_size:(i + 1) * batch_size]

                y_predict = self.__f(x)
                grad = self.__gradient(x, y, y_predict)
                self.w -= lr / np.sqrt(step) * grad

                step += 1

            print(f"epoch {epoch} cross entropy: {self.__cross_entropy(y_train,self.__f(x_train)) / train_size}")

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        # A safe implementation to avoid overflow
        return 0.5 * (1 + np.tanh(0.5 * x))

    def __f(self, x: np.ndarray) -> np.ndarray:
        return self.__sigmoid(np.dot(x, self.w))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.round(self.__f(x)).astype(int)

    def score(self, x_test, y_test):
        return np.mean(y_test == self.predict(x_test))
