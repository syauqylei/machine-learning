from linear_regression.utils import MatrixUtils


def test_create_matrix_x():
    x = [1, 2, 3, 4, 5]

    X = MatrixUtils.generate_x(x)

    for i in range(5):
        for j in range(2):
            if j == 0:
                assert X[i][j] == 1
            if j == 1:
                assert X[i][j] == x[i]
