"""13_debug_features.tsv must hold what the ranker actually saw.

Every offline analysis in PROGRESS.md (F-011, F-013, F-020, F-022, F-030) reads
that table and reasons about the model from it. The heavy-tail log1p transform
used to run *after* the table was written, so five ranker columns were recorded
untransformed: a refit from the table diverged from the run it was reproducing
(kidney's q-value floor came out 0.0222 against the run's 0.0536) and the cause
took a while to find, because the table looked complete and self-consistent.

This is an ordering constraint inside one long function, which is exactly the
shape of bug F-018 records (a guard that ran before the columns it needed
existed). A unit test of any single helper cannot see it, so the check is on the
source order itself.
"""

import inspect

from msi_picasso import pipeline


def test_heavy_tail_transform_precedes_the_debug_write():
    src = inspect.getsource(pipeline.rescore)
    transform = src.index("_HEAVY_TAIL_FEATURES = (")
    # anchor on the write statement, not the filename -- the comment above the
    # transform names the file too
    write = src.index('.to_csv(f"{output_dir}/13_debug_features.tsv"')
    assert transform < write, (
        "the heavy-tail log1p transform moved below the 13_debug_features.tsv "
        "write, so the debug table no longer records what the ranker saw"
    )


def test_transformed_columns_are_all_ranker_eligible():
    """If a heavy-tail column were excluded from the ranker anyway the transform
    would be dead weight; this keeps the list honest as features come and go."""
    src = inspect.getsource(pipeline.rescore)
    block = src[src.index("_HEAVY_TAIL_FEATURES = ("):]
    names = [ln.strip().strip('",') for ln in block.split(")")[0].splitlines()[1:] if ln.strip()]
    assert names, "could not parse _HEAVY_TAIL_FEATURES"
    from msi_picasso.feature_generator import MALDI_INTRINSIC_FEATURES
    for n in names:
        assert n in MALDI_INTRINSIC_FEATURES, f"{n} is transformed but never reaches the ranker"
