import contextlib
import copy
from typing import Iterator

import numpy
import pandas
import pytest

import yosegi
import yosegi.data
import yosegi.fold


@pytest.fixture
def sample_index() -> pandas.Index:
    return pandas.Index(
        ["sample1", "sample2", "sample2", "sample4"]  # intentionally duplicate
    )


@pytest.fixture
def features(sample_index: pandas.Index) -> pandas.DataFrame:
    return pandas.DataFrame(
        {"feature1": [0, 1, 2, 10], "feature2": [3, 4, 5, 20]}, index=sample_index,
    )


@pytest.fixture
def labels(sample_index: pandas.Index) -> pandas.Series:
    return pandas.Series(["Good", "Bad", "Good", "Dunno"], index=sample_index)


@pytest.fixture
def data(features: pandas.DataFrame, labels: pandas.Series) -> yosegi.Data:
    return yosegi.Data(features=features, labels=labels)


@contextlib.contextmanager
def check_immutable(data: yosegi.Data) -> Iterator[None]:
    copied = copy.deepcopy(data)
    feature_id = id(data.features)
    labels_id = id(data.labels)
    label_names_id = id(data.label_names)
    yield
    assert feature_id == id(data.features)
    assert labels_id == id(data.labels)
    assert label_names_id == id(data.label_names)
    assert copied == data


def check_new_instance(before: yosegi.Data, after: yosegi.Data) -> None:
    assert id(before) != id(after)
    assert id(before.features) != id(after.features)
    assert id(before.labels) != id(after.labels)
    assert id(before.label_names) != id(after.label_names)


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
            features=features, labels=labels.reset_index(drop=True),
        )


def test_binarize(data: yosegi.Data) -> None:
    with check_immutable(data):
        binarized_labels = data.binarized_labels
    assert isinstance(binarized_labels, pandas.DataFrame)
    assert binarized_labels.shape == (data.features.shape[0], data.label_names.shape[0])
    assert (binarized_labels.dtypes == numpy.int).all()
    assert (binarized_labels.columns == data.label_names).all()
    assert (binarized_labels.index == data.features.index).all()


def test_label_map_with_reduction(data: yosegi.Data) -> None:
    features_shape = data.features.shape

    with check_immutable(data):
        reduced = data.label_map({"Good": "Better"})
    check_new_instance(data, reduced)
    assert id(reduced) != id(data)
    assert isinstance(reduced, yosegi.Data)
    assert reduced.features.shape == (2, features_shape[1])
    assert reduced.labels.shape == (2,)
    assert (reduced.label_names == ["Better"]).all()


def test_label_map_without_reduction(data: yosegi.Data) -> None:
    features_shape = data.features.shape
    labels_shape = data.labels.shape

    with check_immutable(data):
        mapped = data.label_map({"Good": "Better", "Bad": "Worse", "Dunno": "Unknown"})
    check_new_instance(data, mapped)
    assert id(mapped) != id(data)
    assert isinstance(mapped, yosegi.Data)
    assert mapped.features.shape == features_shape
    assert mapped.labels.shape == labels_shape
    assert (mapped.label_names == ["Better", "Unknown", "Worse"]).all()


def test_label_map_with_error(data: yosegi.Data) -> None:
    with check_immutable(data):
        data.label_map({"Dunno": "Good"})
    with pytest.raises(ValueError):
        data.label_map({"Unknown": "Good"})


def test_reduce_features(data: yosegi.Data) -> None:
    with check_immutable(data):
        reduced = data.reduce_features(None)
    check_new_instance(data, reduced)
    assert isinstance(reduced, yosegi.Data)
    assert reduced.features.shape == (4, 2)

    with check_immutable(data):
        reduced = data.reduce_features(["feature2"])
    check_new_instance(data, reduced)
    assert isinstance(reduced, yosegi.Data)
    assert reduced.features.shape == (4, 1)
    assert (reduced.features.columns == ["feature2"]).all()


def test_create_fold(data: yosegi.Data) -> None:
    # older version of the API
    assert yosegi.data._create_fold(0, 4) == yosegi.fold.Fold(0, 4, 0)
    assert yosegi.data._create_fold(2, 4) == yosegi.fold.Fold(0, 4, 2)
    assert yosegi.data._create_fold(4, 4) == yosegi.fold.Fold(1, 4, 0)
    assert yosegi.data._create_fold(6, 4) == yosegi.fold.Fold(1, 4, 2)
    # fold_idx is recommended as it adds structure
    assert yosegi.data._create_fold(6, 4, 2) == yosegi.fold.Fold(6, 4, 2)


def test_split(data: yosegi.Data) -> None:
    with check_immutable(data):
        train0, test0 = data.split(n_splits=2, fold_idx=0, random_state=0)
        check_new_instance(data, train0)
        check_new_instance(data, test0)
        _, test1 = data.split(n_splits=2, fold_idx=1, random_state=0)
        train2, _ = data.split(n_splits=2, fold_idx=0, random_state=1)

    assert set(train0.index.tolist() + test0.index.tolist()) == set(data.index)
    assert set(train0.index) == set(test1.index)
    assert set(train0.index) != set(train2.index)
    assert set(test0.index) != set(test1.index)


def test_to_dataframe(data: yosegi.Data) -> None:
    with check_immutable(data):
        df = data.to_dataframe()
    original_shape = data.features.shape
    assert df.shape == (original_shape[0], original_shape[1] + 1)
    assert (df.columns == ["feature1", "feature2", "labels"]).all()
