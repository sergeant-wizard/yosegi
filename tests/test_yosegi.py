import copy

import numpy
import pandas
import pytest

import yosegi


@pytest.fixture
def sample_index() -> pandas.Index:
    yield pandas.Index([
        'sample1',
        'sample2',
        'sample3',
        'sample4',
    ])


@pytest.fixture
def features(sample_index) -> pandas.DataFrame:
    yield pandas.DataFrame({
        'feature1': [0, 1, 2, 10],
        'feature2': [3, 4, 5, 20],
    },
        index=sample_index,
    )


@pytest.fixture
def labels(sample_index) -> pandas.Series:
    yield pandas.Series(
        ['Good', 'Bad', 'Good', 'Dunno'],
        index=sample_index,
    )


@pytest.fixture
def data(features, labels) -> yosegi.Data:
    yield yosegi.Data(
        features=features,
        labels=labels,
    )


def test_good_data(data: yosegi.Data) -> None:
    assert isinstance(data.features, pandas.DataFrame)
    assert data.features.shape == (4, 2)

    assert isinstance(data.labels, pandas.Series)
    assert data.labels.shape == (4,)

    assert isinstance(data.label_names, numpy.ndarray)
    assert data.label_names.shape == (3,)
    assert data.label_names.dtype == object

    assert isinstance(data.index, pandas.Index)
    assert (data.index == data.features.index).all()
    assert (data.index == data.labels.index).all()

    assert (data.features.index == data.labels.index).all()


def test_equals(data: yosegi.Data) -> None:
    copied = copy.deepcopy(data)
    assert data == copied
    copied.features.iloc[0, 0] += 1
    assert not data == copied


def test_bad_data(features: pandas.DataFrame, labels: pandas.Series) -> None:
    with pytest.raises(AssertionError):
        yosegi.Data(
            features=features,
            labels=labels.reset_index(drop=True),
        )


def test_binarize(data) -> None:
    binarized_labels = data.binarized_labels
    assert isinstance(binarized_labels, pandas.DataFrame)
    assert binarized_labels.shape == (
        data.features.shape[0], data.label_names.shape[0])
    assert (binarized_labels.dtypes == numpy.int).all()
    assert (binarized_labels.columns == data.label_names).all()
    assert (binarized_labels.index == data.features.index).all()


def test_label_map_with_reduction(data) -> None:
    features_shape = data.features.shape

    data = data.label_map({
        'Good': 'Better',
    })
    assert isinstance(data, yosegi.Data)
    assert data.features.shape == (2, features_shape[1])
    assert data.labels.shape == (2,)
    assert (data.label_names == ['Better']).all()


def test_label_map_without_reduction(data) -> None:
    features_shape = data.features.shape
    labels_shape = data.labels.shape

    data = data.label_map({
        'Good': 'Better',
        'Bad': 'Worse',
        'Dunno': 'Unknown',
    })
    assert isinstance(data, yosegi.Data)
    assert data.features.shape == features_shape
    assert data.labels.shape == labels_shape
    assert (data.label_names == ['Better', 'Unknown', 'Worse']).all()


def test_reduce_features(data) -> None:
    data = data.reduce_features(None)
    assert isinstance(data, yosegi.Data)
    assert data.features.shape == (4, 2)

    data = data.reduce_features(['feature2'])
    assert isinstance(data, yosegi.Data)
    assert data.features.shape == (4, 1)
    assert (data.features.columns == ['feature2']).all()


def test_split(data) -> None:
    train0, test0 = data.split(n_splits=2, random_state=0)
    train1, test1 = data.split(n_splits=2, random_state=1)

    assert set(train0.index.tolist() + test0.index.tolist()) == set(data.index)
    assert set(train0.index) == set(test1.features.index)
