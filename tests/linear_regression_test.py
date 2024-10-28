import pandas as pd
import os
import pytest

from linear_regression.main import LinearRegression


@pytest.fixture
def dataset():
    dataset = pd.read_csv(f"{os.getcwd()}/tests/AirQualityUCI.csv")
    dataset = dataset.drop([x for x in dataset.columns.values[-2:]], axis=1)
    dataset = dataset.dropna()
    return dataset


def test_linear_regression(dataset):
    reg = LinearRegression()

    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    reg.fit(x, y)

    assert reg.XtX is not None
    assert reg.Xty is not None
    assert reg.BETAS is not None

    assert reg.BETAS[0] == 0
    assert reg.BETAS[1] == 1

    y_predict = reg.predict(6)

    assert y_predict == 6

    y_predict = reg.predict(7)

    assert y_predict == 7
