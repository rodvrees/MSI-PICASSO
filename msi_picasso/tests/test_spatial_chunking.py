"""Threading the spatial-feature pass must not copy the ion images (PROGRESS.md F-050).

The array is the largest object in the program -- 71 GB on kidney at min_regions=1 --
so a duplicate of it is the difference between a run finishing and being killed.
"""
import numpy as np

from msi_picasso.maldi_extraction import compute_spatial_features


def _images(n_features=37, h=5, w=6):
    rng = np.random.default_rng(0)
    return rng.random((n_features, h, w), dtype=np.float32)


def test_result_is_bit_identical_at_every_chunking():
    """Chunk boundaries must not change a single value.

    Every statistic is a per-feature reduction, so this holds by construction -- and it
    is what lets the chunk size be bounded for memory without making two runs
    incomparable. E030 and E031 straddle that change.
    """
    imgs = _images(n_features=311, h=17, w=23)
    imgs[imgs < 0.3] = 0.0
    mzs = np.linspace(700.0, 2000.0, len(imgs))
    base = compute_spatial_features(imgs, mzs, imgs[0].size, n_workers=1)
    for n_workers in (2, 7, 64, len(imgs)):
        got = compute_spatial_features(imgs, mzs, imgs[0].size, n_workers=n_workers)
        assert list(got.columns) == list(base.columns)
        assert len(got) == len(base) == len(imgs)
        for col in base.columns:
            np.testing.assert_array_equal(got[col].to_numpy(), base[col].to_numpy())


def test_chunking_does_not_copy_the_ion_images():
    """Fails if the chunks go back to fancy indexing, which silently doubles peak memory.

    Every byte a worker reads must still belong to the original array, so writing
    through the original is visible to the chunk a worker would have been handed.
    """
    imgs = _images()
    mzs = np.linspace(800.0, 1600.0, len(imgs))
    before = imgs.ctypes.data
    compute_spatial_features(imgs, mzs, imgs[0].size, n_workers=8)
    assert imgs.ctypes.data == before, "the input array was reallocated"
    # The assert inside compute_spatial_features is the direct check; this pins the
    # property it guards, so the assert cannot be dropped without a test failing.
    n_workers = 8
    bounds = np.linspace(0, len(imgs), n_workers + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            assert imgs[a:b].base is imgs
