import dataclasses


@dataclasses.dataclass
class Fold:
    random_state: int
    n_splits: int
    fold_idx: int

    @classmethod
    def from_flat_index(cls, idx: int, n_splits: int) -> 'Fold':
        return cls(
            fold_idx=idx % n_splits,
            random_state=idx // n_splits,
            n_splits=n_splits,
        )
