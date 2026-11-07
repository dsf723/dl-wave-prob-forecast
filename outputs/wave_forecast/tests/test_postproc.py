"""Checks for EMOS closed-form CRPS, the Hersbach decomposition, conditional PIT and conformal shifts."""
import unittest

import numpy as np

from wavecast.metrics import TAU, score
from wavecast.postproc import (crps_tn, crps_ln, emos_fit, emos_quantiles, hersbach, conditional_pit,
                               conformal_shifts, apply_shifts, pit)


class PostprocTests(unittest.TestCase):
    def test_closed_form_crps_matches_quantile_grid(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2., .7, 200)
        mean, spread, g = np.full(200, 1.4), np.full(200, .5), np.array(["s|1"] * 200)
        for kind, f in (("tn", crps_tn), ("ln", crps_ln)):
            th = {"s|1": np.array([0., 1., .3, 1.])}
            q = emos_quantiles(th, mean, spread, g, kind)
            from wavecast.postproc import _params
            p1, p2 = _params(kind, th["s|1"], mean, spread)
            # the 99-point grid approximates the continuous CRPS to within about 1 %
            self.assertAlmostEqual(score(q, y).mean() / f(p1, p2, y).mean(), 1., delta=.02)

    def test_emos_recovers_simple_relation(self):
        rng = np.random.default_rng(1)
        m = rng.uniform(.5, 3, 4000); s = rng.uniform(.1, .5, 4000)
        y = np.abs(.2 + .9 * m + rng.normal(0, 1, 4000) * np.sqrt(.04 + s**2))
        th = emos_fit(m, s, y, np.array(["a"] * 4000), "tn")["a"]
        self.assertAlmostEqual(th[1], .9, delta=.05)

    def test_hersbach_identity(self):
        rng = np.random.default_rng(2)
        y = rng.gamma(2., .7, 500)
        q = np.sort(rng.gamma(2., .7, (500, 99)) * rng.uniform(.6, 1.4, (500, 1)), axis=1)
        d = hersbach(q, y)
        self.assertAlmostEqual(d["crps"], score(q, y).mean(), places=6)
        self.assertAlmostEqual(d["crps"], d["reliability"] + d["potential"], places=6)
        self.assertGreaterEqual(d["reliability"], 0)

    def test_pit_uniform_for_calibrated_forecast(self):
        rng = np.random.default_rng(3)
        y = rng.normal(5, 1, 5000)
        q = np.tile(5 + np.sqrt(2) * __import__("scipy").special.erfinv(2 * TAU - 1), (5000, 1))
        h = np.histogram(pit(q, y, rng), bins=10, range=(0, 1))[0] / 5000
        self.assertLess(np.abs(h - .1).max(), .02)
        c, k = conditional_pit(q, y, 6., rng)
        h = np.histogram(c, bins=5, range=(0, 1))[0] / k.sum()
        self.assertLess(np.abs(h - .2).max(), .04)

    def test_conformal_shift_restores_coverage(self):
        rng = np.random.default_rng(4)
        y = rng.normal(0, 1, 4000) + 3
        q = np.tile(3 + .6 * np.sqrt(2) * __import__("scipy").special.erfinv(2 * TAU - 1), (4000, 1))  # too narrow
        g = np.array(["a"] * 4000)
        sh = conformal_shifts(q[:2000], y[:2000], g[:2000])
        q2 = apply_shifts(q[2000:], g[2000:], sh)
        cover = ((y[2000:] >= q2[:, 9]) & (y[2000:] <= q2[:, 89])).mean()
        self.assertAlmostEqual(cover, .8, delta=.03)
        self.assertTrue((np.diff(q2, axis=1) >= 0).all())


if __name__ == "__main__":
    unittest.main()
