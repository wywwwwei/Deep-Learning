import numpy as np


class LinearRegression:

    def __init__(self) -> None:
        super().__init__()

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, epochs: int = 1000, lr: float = 100.0) -> None:
        dimension = train_x.shape[1]
        self.w = np.zeros([dimension, 1])

        adagrad = np.zeros([dimension, 1])
        eps = 0.0000000001  # avoid adagrad = 0
        sample_num = train_x.shape[0]
        for i in range(epochs):
            predict_y = self.predict(train_x)
            diff = predict_y - train_y
            loss = np.sqrt(np.sum(np.power(diff, 2)) / sample_num)
            if i % 100 == 0:
                print(f"epoch {i} loss: {loss}")
            gradient = 2 * np.dot(train_x.transpose(), diff)
            adagrad += gradient**2
            self.w -= lr * gradient / np.sqrt(adagrad + eps)

    def predict(self, x: np.ndarray):
        return np.dot(x, self.w)

    def score(self, test_x: np.ndarray, test_y: np.ndarray, precision: float = 0) -> float:
        return np.mean(abs(test_y - self.predict(test_x)) <= precision)