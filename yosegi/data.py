import pathlib
import typing

import numpy
import pandas
import sklearn.model_selection

import yosegi.io
from yosegi.fold import Fold


def _create_fold(
    random_state: int,
    n_splits: int,
    fold_idx: typing.Optional[int] = None,
) -> Fold:
    """
    support older versions of Data.split method
    """
    if fold_idx is None:
        return Fold.from_flat_index(
            idx=random_state,
            n_splits=n_splits,
        )
    else:
        return Fold(
            fold_idx=fold_idx,
            random_state=random_state,
            n_splits=n_splits,
        )


class Data:
    def __init__(
        self,
        features: pandas.DataFrame,
        labels: pandas.Series,
    ) -> None:
        assert (features.index == labels.index).all()
        self._features = features
        self._labels = labels
        self._label_names = numpy.unique(self._labels)

    @property
    def features(self) -> pandas.DataFrame:
        return self._features.copy()

    @property
    def labels(self) -> pandas.Series:
        return self._labels.copy()

    @property
    def label_names(self) -> numpy.ndarray:
        return self._label_names.copy()

    @classmethod
    def from_dataframe(cls, df: pandas.DataFrame) -> 'Data':
        return cls(
            features=df.loc[:, df.columns != 'labels'],
            labels=df['labels'],
        )

    def to_dataframe(self) -> pandas.DataFrame:
        assert 'labels' not in self._features.columns
        ret = self._features.copy()
        ret['labels'] = self._labels
        return ret

    def __eq__(self, other: object) -> bool:
        if isinstance(other, yosegi.Data):
            return all([
                (self._features == other._features).all().all(),
                (self._labels == other._labels).all(),
                (self._label_names == other._label_names).all(),
            ])
        else:
            raise NotImplementedError

    @property
    def binarized_labels(self) -> pandas.DataFrame:
        label_set = numpy.unique(self._labels)
        labels = sklearn.preprocessing.label_binarize(
            self._labels,
            label_set,
        )
        labels = pandas.DataFrame(
            labels,
            index=self._features.index,
            columns=label_set,
        )
        return labels

    @property
    def index(self) -> pandas.Index:
        return self._features.index

    def label_map(self, mapping: dict) -> 'Data':
        if not set(mapping.keys()).issubset(self._labels.unique()):
            raise ValueError('At least one key was missing in labels')

        valid_index = numpy.isin(
            self._labels, list(mapping.keys())
        )
        labels = self._labels[valid_index]
        labels = labels.apply(mapping.get)
        features = self._features[valid_index]

        return Data(
            features=features,
            labels=labels,
        )

    def reduce_features(
        self,
        feature_idx: typing.Optional[numpy.array],
    ) -> 'Data':
        if feature_idx is None:
            features = self._features.copy()
        else:
            features = self._features[feature_idx]
        return Data(
            features=features,
            labels=self._labels.copy(),
        )

    def save(
        self,
        path: pathlib.Path,
        fmt: yosegi.io.Formats = yosegi.io.Formats.joblib,
    ) -> None:
        fmt.formatter.save(self, path)

    @staticmethod
    def load(
        path: pathlib.Path,
        fmt: yosegi.io.Formats = yosegi.io.Formats.joblib,
    ) -> 'Data':
        return fmt.formatter.load(path)

    @classmethod
    def merge(
        cls,
        *datas: 'Data',
    ) -> 'Data':
        return Data(
            features=pandas.concat([data._features for data in datas]),
            labels=pandas.concat([data._labels for data in datas]),
        )

    def split(
        self,
        random_state: int,
        n_splits: int,
        fold_idx: typing.Optional[int] = None,
    ) -> typing.Tuple['Data', 'Data']:
        fold = _create_fold(random_state, n_splits, fold_idx)
        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=fold.n_splits,
            random_state=fold.random_state,
            shuffle=True,
        )
        if self._labels.ndim == 1:
            labels = self._labels
        else:
            labels = numpy.argmax(self._labels.values, axis=1)
        train_index, test_index = list(
            skf.split(self._features, labels)
        )[fold.fold_idx]
        return (
            Data(
                features=self._features.iloc[train_index],
                labels=self._labels.iloc[train_index],
            ),
            Data(
                features=self._features.iloc[test_index],
                labels=self._labels.iloc[test_index],
            ),
        )
