"""Tests for the leakage-free participant-level splitter."""

import pandas as pd
import pytest

from src.data.splitter import LeakageFreeSplitter, add_corpus_id


def _toy_manifest(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "participant_id": f"p{i}",
                "corpus": ["daic", "eatd", "androids"][i % 3],
                "label": i % 2,
            }
        )
    return add_corpus_id(pd.DataFrame(rows))


def test_no_participant_leakage_across_test_folds():
    df = _toy_manifest()
    splitter = LeakageFreeSplitter(df, n_folds=5, seed=0, mode="pooled")
    seen = set()
    for _, test_ids in splitter.folds():
        assert not seen.intersection(test_ids), "participant appears in two test folds"
        seen.update(test_ids)
    # Every participant is tested exactly once.
    assert seen == set(df["participant_id"])


def test_train_test_disjoint():
    df = _toy_manifest()
    splitter = LeakageFreeSplitter(df, n_folds=5, seed=1)
    for train_ids, test_ids in splitter.folds():
        assert set(train_ids).isdisjoint(test_ids)


def test_assert_no_leakage_passes():
    df = _toy_manifest()
    LeakageFreeSplitter(df, n_folds=5, seed=2).assert_no_leakage()


def test_loco_holds_out_one_corpus():
    df = _toy_manifest()
    splitter = LeakageFreeSplitter(df, mode="loco")
    n = 0
    for train_ids, test_ids in splitter.folds():
        train_corpora = set(df.set_index("participant_id").loc[train_ids, "corpus"])
        test_corpora = set(df.set_index("participant_id").loc[test_ids, "corpus"])
        assert len(test_corpora) == 1
        assert test_corpora.isdisjoint(train_corpora)
        n += 1
    assert n == df["corpus"].nunique()
