import numpy as np


class MatrixUtils:

    @staticmethod
    def generate_x(x: list) -> np.ndarray:
        count = len(x)
        X = np.array([[1, x[i]] for i in range(count)])

        return X
