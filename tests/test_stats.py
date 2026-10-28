"""Known-answer checks for the statistics in analysis_v3. Run: python tests/test_stats.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from analysis_v3 import Bootstrap, _p_two_sided, auroc, holm  # noqa: E402


def test_p_value():
    # no resample crosses zero: k = 0, p = 2/(B+1)
    assert np.isclose(_p_two_sided(np.full(1000, 0.01)), 2 / 1001)
    # 30 of 1000 at or below zero: p = 2 * 31 / 1001
    d = np.r_[np.full(30, -0.01), np.full(970, 0.01)]
    assert np.isclose(_p_two_sided(d), 62 / 1001)
    # symmetric around zero: capped at 1
    assert _p_two_sided(np.r_[np.full(500, -1.0), np.full(500, 1.0)]) == 1.0
    # NaN resamples are dropped
    assert np.isclose(_p_two_sided(np.r_[np.full(999, 0.01), np.nan]), 2 / 1000)


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])  # sorted: 0.01*3, 0.03*2, then max(0.06, 0.04*1)
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_bootstrap_keeps_patients_together():
    groups = np.array([1, 1, 1, 2, 3, 3])
    bs = Bootstrap(groups, n=200, seed=0)
    for idx in bs.idx:
        picked = groups[idx]
        for g in np.unique(picked):
            # every draw of a patient brings all of that patient's images
            assert (picked == g).sum() % (groups == g).sum() == 0


def test_auroc():
    y = np.array([[0], [0], [1], [1]])
    assert np.isclose(auroc(y, np.array([[0.1], [0.2], [0.8], [0.9]]))[0], 1.0)
    assert np.isclose(auroc(y, np.array([[0.9], [0.8], [0.2], [0.1]]))[0], 0.0)
    assert np.isclose(auroc(y, np.array([[0.5], [0.5], [0.5], [0.5]]))[0], 0.5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
