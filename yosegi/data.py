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
        self.features = features
        self.labels = labels
        self.label_names = numpy.unique(self.labels)

    @classmethod
    def from_dataframe(cls, df: pandas.DataFrame) -> 'Data':
        return cls(
            features=df.loc[:, df.columns != 'labels'],
            labels=df['labels'],
        )

    def to_dataframe(self) -> pandas.DataFrame:
        assert 'labels' not in self.features.columns
        ret = self.features.copy()
        ret['labels'] = self.labels
        return ret

    def __eq__(self, other) -> bool:
        return all([
            (self.features == other.features).all().all(),
            (self.labels == other.labels).all(),
            (self.label_names == other.label_names).all(),
        ])

    @property
    def binarized_labels(self) -> pandas.DataFrame:
        label_set = numpy.unique(self.labels)
        labels = sklearn.preprocessing.label_binarize(
            self.labels,
            label_set,
        )
        labels = pandas.DataFrame(
            labels,
            index=self.features.index,
            columns=label_set,
        )
        return labels

    @property
    def index(self) -> pandas.Index:
        return self.features.index

    def label_map(
        self,
        mapping: typing.Dict[str, typing.Union[str, int]],
    ) -> 'Data':
        if not set(mapping.keys()).issubset(self.labels.unique()):
            raise ValueError('At least one key was missing in labels')

        valid_index = numpy.isin(
            self.labels, list(mapping.keys())
        )
        labels = self.labels[valid_index]
        labels = labels.apply(mapping.get)
        features = self.features[valid_index]

        return Data(
            features=features,
            labels=labels,
        )

    def reduce_features(
        self,
        feature_idx: typing.Optional[numpy.array],
    ) -> 'Data':
        if feature_idx is None:
            features = self.features.copy()
        else:
            features = self.features[feature_idx]
        return Data(
            features=features,
            labels=self.labels.copy(),
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
            features=pandas.concat([data.features for data in datas]),
            labels=pandas.concat([data.labels for data in datas]),
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
        if self.labels.ndim == 1:
            labels = self.labels
        else:
            labels = numpy.argmax(self.labels.values, axis=1)
        train_index, test_index = list(
            skf.split(self.features, labels)
        )[fold.fold_idx]
        return (
            Data(
                features=self.features.iloc[train_index],
                labels=self.labels.iloc[train_index],
            ),
            Data(
                features=self.features.iloc[test_index],
                labels=self.labels.iloc[test_index],
            ),
        )
